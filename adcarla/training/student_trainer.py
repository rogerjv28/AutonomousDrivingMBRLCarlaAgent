"""Etapa 2: entrenamiento del alumno (visión o fusión) guiado por el profesor. Requiere CARLA.

El profesor guía al alumno de tres formas (Raw2Drive 3.3), una por hook de `StudentTrainer`:
  - `compute_wm_loss()`: Rollout Guidance. Compara la rejilla BEV, el estado estocástico y `h`
    del alumno con los del profesor sobre el mismo batch. El alumno no entrena sus propias
    cabezas reward/cont.
  - `get_reward_fn()`, `get_cont_fn()` y `get_teacher()`: Head Guidance. La imaginación usa las
    cabezas reward/cont del profesor y destila su política en la del alumno.
  - `collect_policy()`: el profesor conduce cada episodio con probabilidad `p_teacher`, que
    decae de 1 a 0 durante el entrenamiento.

El alumno parte de inicialización propia (sin heredar encoder del profesor ni de la otra rama)
para una comparativa justa. El RSSM (`init_rssm_from_teacher`) y el actor y el critic
(`init_actor_critic_from_teacher`) sí pueden partir de los pesos del profesor.
"""
import random

import torch

from adcarla.carla_env.env import CarlaEnv
from adcarla.encoders.factory import build_encoder
from adcarla.guidance.guidance import HeadGuidance, RolloutGuidance, teacher_probability
from adcarla.policy.factory import build_actor, build_critic
from adcarla.training.agent import RolloutAgent, collect_episode, flatten_states
from adcarla.training.base_trainer import BaseTrainer
from adcarla.world_model.world_model import WorldModel


class StudentTrainer(BaseTrainer):
    """Entrena el world model y la política del alumno guiado por el profesor (Etapa 2)."""

    def __init__(self, config: dict, env=None, init_rssm_from_teacher: bool = True,
                 init_actor_critic_from_teacher: bool = True):
        """
        Args:
            config: configuración completa (config["branch"] identifica "vision" o "fusion").
            env: entorno ya construido (opcional).
            init_rssm_from_teacher: si es True, preinicializa el RSSM del alumno desde el
                profesor para acelerar la convergencia (el encoder y las cabezas siguen siendo propios).
            init_actor_critic_from_teacher: si es True, el actor-critic del alumno parten de
                los pesos del profesor en vez de una inicialización aleatoria.
        """
        self.init_rssm_from_teacher = init_rssm_from_teacher
        self.init_actor_critic_from_teacher = init_actor_critic_from_teacher
        super().__init__(config, env=env)
        guidance_config = self.config.get("guidance", {}) or {}
        self.p_teacher_fraction = float(guidance_config["p_teacher_fraction"])
        # Generador propio con semilla (como TrafficManager y RouteManager) que decide qué episodios
        # conduce el profesor. Así visión y fusión coinciden en ellos y la comparación sigue siendo justa.
        self._rng = random.Random(int(self.config.get("seed", 0)))

    def _build_env(self):
        self.config = {**self.config, "privileged_bev": True}   # target del decoder
        return CarlaEnv(self.config)

    def _load_teacher(self):
        """Carga el world model, el actor y el critic del profesor desde `train.teacher_checkpoint` y los congela.

        El actor se usa para la recolección mixta y la destilación; el critic, para copiarlo
        al alumno cuando `init_actor_critic_from_teacher` está activo. Los tres salen del
        checkpoint que guarda `BaseTrainer.save_checkpoint()`.
        """
        checkpoint_path = self.train_config["teacher_checkpoint"]
        teacher_config = {**self.config, "encoder": "privileged"}
        teacher_wm = WorldModel(teacher_config, build_encoder(teacher_config), self.num_actions).to(self.device)
        teacher_actor = build_actor(self.config, teacher_wm.rssm.feat_dim, self.num_actions).to(self.device)
        teacher_critic = build_critic(self.config, teacher_wm.rssm.feat_dim).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "actor" not in checkpoint:
            raise RuntimeError(
                f"el checkpoint del profesor ({checkpoint_path}) no trae 'actor': la recolección "
                "mixta y la destilación necesitan la política del profesor, no solo su world model")
        if "critic" not in checkpoint:
            raise RuntimeError(
                f"el checkpoint del profesor ({checkpoint_path}) no trae 'critic': "
                "se necesita también su critic")
        teacher_wm.load_state_dict(checkpoint["wm"])
        teacher_actor.load_state_dict(checkpoint["actor"])
        teacher_critic.load_state_dict(checkpoint["critic"])

        for module in (teacher_wm, teacher_actor, teacher_critic):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)   # el profesor se usa solo como referencia
        return teacher_wm, teacher_actor, teacher_critic

    def _build_models(self):
        self.teacher_wm, self.teacher_actor, self.teacher_critic = self._load_teacher()   # congelados: solo guidance

        student_wm = WorldModel(self.config, build_encoder(self.config), self.num_actions).to(self.device)
        if self.init_rssm_from_teacher:
            student_wm.rssm.load_state_dict(self.teacher_wm.rssm.state_dict())

        # Va dentro del world model del alumno para que su proyección 1x1 (si la hay) se optimice
        # con `opt_wm` y se guarde en el checkpoint.
        student_wm.rollout_guidance = RolloutGuidance(
            self.config, student_wm.encoder.grid_channels, self.teacher_wm.encoder.grid_channels).to(self.device)

        actor = build_actor(self.config, student_wm.rssm.feat_dim, self.num_actions).to(self.device)
        critic = build_critic(self.config, student_wm.rssm.feat_dim).to(self.device)
        if self.init_actor_critic_from_teacher:
            # Las dimensiones salen de la misma configuración que el profesor, así que las
            # arquitecturas coinciden aunque no se copie el RSSM.
            actor.load_state_dict(self.teacher_actor.state_dict())
            # El state_dict del critic incluye el target lento (`slow`): cargarlo entero evita que
            # `slow` siga siendo una copia de la red aleatoria mientras `net` ya es la del profesor.
            critic.load_state_dict(self.teacher_critic.state_dict())
        self.head_guidance = HeadGuidance(self.config, self.teacher_wm, self.teacher_actor)
        agent = RolloutAgent(student_wm, actor, self.config, self.device)
        # El profesor necesita el BEV privilegiado para conducir. El entorno lo sigue
        # entregando (`privileged_bev` en `_build_env`).
        self.teacher_agent = RolloutAgent(self.teacher_wm, self.teacher_actor, self.config, self.device)
        return student_wm, actor, critic, agent

    def compute_wm_loss(self, batch: dict):
        """World model del alumno (solo recon+kl) + Rollout Guidance contra el profesor congelado."""
        aux = {}   # `WorldModel.loss` guarda aquí la rejilla BEV y los logits posteriores del alumno
        wm_loss, states, metrics = self.wm.loss(batch, terms=("recon", "kl"), aux=aux)

        with torch.no_grad():
            teacher_embed, teacher_grid = self.teacher_wm._encode_seq(batch, with_grid=True)
            # El profesor no muestrea, usa la muestra del alumno en cada paso.
            # Así los tres términos comparan estados y no ruido de muestreo.
            teacher_states, teacher_post, _, _ = self.teacher_wm.rssm.observe(
                teacher_embed, batch["prev_action"], stoch_override=states["stoch"])

        # La imaginación arranca de `flatten_states(states)`: el profesor debe arrancar de su
        # propio estado en esos mismos instantes para avanzar en paralelo con el alumno.
        self.head_guidance.reset(flatten_states(teacher_states))

        guidance_loss, guidance_metrics = self.wm.rollout_guidance(
            student={"grid": aux["bev_grid"], "post_logits": aux["post_logits"], "h": states["h"]},
            teacher={"grid": teacher_grid, "post_logits": teacher_post, "h": teacher_states["h"]})

        metrics.update(guidance_metrics)
        total_loss = wm_loss + guidance_loss
        metrics["loss"] = total_loss.item()
        return total_loss, states, metrics

    def get_reward_fn(self):
        return self.head_guidance.reward_fn

    def get_cont_fn(self):
        return self.head_guidance.cont_fn

    def get_teacher(self):
        """Solo es válido tras `compute_wm_loss()`, que hace el `reset()` con el estado del
        profesor del batch en curso. El bucle de `BaseTrainer` los llama en ese orden."""
        return self.head_guidance

    def collect_policy(self, episode: int) -> int:
        """Recoge un episodio conduciendo el profesor con probabilidad `p_teacher`.

        Args:
            episode: episodio actual. `p_teacher` decae linealmente de 1 a 0 con él.

        Returns:
            Pasos actuados en el episodio (los cuenta `collect_episode`).
        """
        p_teacher = teacher_probability(episode, int(self.train_config["total_episodes"]),
                                        self.p_teacher_fraction)
        agent = self.teacher_agent if self._rng.random() < p_teacher else self.agent
        return collect_episode(self.env, agent, self.buffer, self.config, self.num_actions, greedy=False)


def train_student(config: dict, resume: bool = False, init_rssm_from_teacher: bool = True,
                   init_actor_critic_from_teacher: bool = True):
    """Punto de entrada de scripts/train_student.py.

    Args:
        config: configuración completa (ver configs/student_vision.yaml o student_fusion.yaml).
        resume: si es True, reanuda desde el último checkpoint guardado. Los pesos del checkpoint
            sustituyen a los copiados del profesor, porque se cargan después de construir los modelos.
        init_rssm_from_teacher: si es True, el RSSM del alumno parte de los pesos del profesor.
        init_actor_critic_from_teacher: si es True, el actor y el critic del alumno parten de los del profesor.

    Returns:
        Tupla (student_wm, actor) con los módulos entrenados.
    """
    trainer = StudentTrainer(config, init_rssm_from_teacher=init_rssm_from_teacher,
                              init_actor_critic_from_teacher=init_actor_critic_from_teacher)
    trainer.train(resume=resume)
    return trainer.wm, trainer.actor
