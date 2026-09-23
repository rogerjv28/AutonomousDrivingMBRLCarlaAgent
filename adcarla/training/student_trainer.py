"""Etapa 2: entrenamiento del alumno (visión o fusión) guiado por el profesor. Requiere CARLA.

El profesor guía al alumno de dos formas (Raw2Drive 3.3):
  - Rollout Guidance: MSE entre estados latentes del profesor y del alumno.
  - Head Guidance: las cabezas de reward/cont del profesor supervisan la política del alumno.

El alumno parte de inicialización propia (sin heredar encoder del profesor ni de la otra rama)
para una comparativa justa. Solo el RSSM puede hacer un "warm-start" desde el profesor.
"""
import torch
import torch.nn.functional as F

from adcarla.carla_env.env import CarlaEnv
from adcarla.encoders.factory import build_encoder
from adcarla.world_model.world_model import WorldModel
from adcarla.policy.actor_critic import Actor, Critic
from adcarla.policy.imagination import imagine_losses
from adcarla.training.replay_buffer import SequenceReplayBuffer
from adcarla.training.agent import RolloutAgent, obs_to_step, flatten_states
from adcarla.guidance.guidance import rollout_guidance_loss, HeadGuidance
from adcarla.utils.distributions import categorical_kl_balance


def _load_teacher(config: dict, num_actions: int, device: str):
    """Carga el World Model del profesor desde checkpoints/teacher.pt y congela sus pesos.

    Args:
        config: configuración base (se sobreescribe el encoder a "privileged").
        num_actions: número de acciones (debe coincidir con el checkpoint).
        device: dispositivo donde cargar el modelo.

    Returns:
        World Model del profesor congelado y en modo eval.
    """
    teacher_config = {**config, "encoder": "privileged"}
    teacher_wm = WorldModel(teacher_config, build_encoder(teacher_config), num_actions).to(device)
    teacher_checkpoint = torch.load("checkpoints/teacher.pt", map_location=device)
    teacher_wm.load_state_dict(teacher_checkpoint["wm"])
    teacher_wm.eval()  # desactiva el dropout y el batch normalization

    for p in teacher_wm.parameters():
        p.requires_grad_(False)     # congela todos los pesos: el profesor se usa solo como referencia

    return teacher_wm


def train_student(config: dict, init_rssm_from_teacher: bool = True):
    """Entrena el world model y la política del alumno guiado por el profesor.

    Args:
        config: configuración completa (ver configs/student_vision.yaml o student_fusion.yaml).
            config["branch"] identifica la rama ("vision" o "fusion").
        init_rssm_from_teacher: si es True, preinicializa el RSSM del alumno desde el profesor
            para acelerar la convergencia (los encoders y heads siguen siendo propios).

    Returns:
        Tupla (student, actor). Guarda checkpoints/student_{branch}.pt.
    """
    device = config.get("device", "cpu")
    config = {**config, "privileged_bev": True}   # el entorno CARLA provee la máscara BEV privilegiada como target del decoder
    train_config = config["train"]

    # --- Inicialización ---
    env = CarlaEnv(config)
    num_actions = env.actions.n
    teacher_wm = _load_teacher(config, num_actions, device)   # congelado: solo para guidance

    student_wm = WorldModel(config, build_encoder(config), num_actions).to(device)
    if init_rssm_from_teacher:
        # Preinicialización de la dinámica temporal, el encoder (la variable a comparar) es propio
        student_wm.rssm.load_state_dict(teacher_wm.rssm.state_dict())

    actor  = Actor(student_wm.rssm.feat_dim, num_actions).to(device)
    critic = Critic(student_wm.rssm.feat_dim, int(config["world_model"]["num_bins"])).to(device)
    head_guidance = HeadGuidance(teacher_wm)   # envuelve reward/cont del profesor como callables

    buffer = SequenceReplayBuffer(int(train_config["replay_capacity"]), int(train_config["seq_len"]))
    agent  = RolloutAgent(student_wm, actor, config, device)

    # Optimizadores separados: actor_loss no puede actualizar critic.net (ni viceversa)
    opt_wm    = torch.optim.Adam(student_wm.parameters(), lr=float(train_config["lr"]))
    opt_actor  = torch.optim.Adam(actor.parameters(),  lr=float(train_config["lr"]))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=float(train_config["lr"]))

    # Ponemos los tres módulos del agente estudiante en modo train antes de entrenar.
    student_wm.train()
    actor.train()
    critic.train()

    # --- Bucle de entrenamiento ---
    episode = 0  # inicialización
    try:
        for episode in range(int(train_config["total_episodes"])):

            observation = env.reset()
            agent.reset()
            done = False

            # Recopilamos información del episodio con los sensores del alumno
            while not done:
                action = agent.act(observation, greedy=False)
                next_obs, reward, done, _ = env.step(action)
                step_data = obs_to_step(observation, config)
                step_data.update(action=action, reward=reward, cont=0.0 if done else 1.0)
                buffer.add_step(step_data)
                observation = next_obs

            buffer.end_episode()

            if not buffer.can_sample():
                continue

            batch = buffer.sample(int(train_config["batch_size"]), device)
            B, T = batch["action"].shape[:2]
            action_onehot = F.one_hot(batch["action"].long(), num_actions).float()  # [B, T, num_actions]

            # --- World Model del alumno: recon + KL + Rollout Guidance ---
            student_embed = student_wm._encode_seq(batch)          # [B, T, embed_dim]
            student_states, post_logits, prior_logits, _ = student_wm.rssm.observe(student_embed, action_onehot)
            student_feat = student_wm.rssm.feat(student_states)    # [B, T, feat_dim]

            # Reconstrucción BEV: aplana (B,T) para el decoder convolucional y recupera la forma
            bev_recon = student_wm.decoder(student_feat.reshape(B * T, -1)).reshape(
                B, T, student_wm.bev_channels, student_wm.size, student_wm.size)
            recon_loss = F.binary_cross_entropy(bev_recon, batch["bev"])

            kl_loss = categorical_kl_balance(post_logits, prior_logits, student_wm.free_bits, student_wm.kl_balance).mean()

            # Rollout Guidance: MSE entre estados latentes, el estado del profesor se evalúa congelado
            with torch.no_grad():
                teacher_embed = teacher_wm._encode_seq(batch)
                teacher_states, _, _, _ = teacher_wm.rssm.observe(teacher_embed, action_onehot)
            guidance_loss = rollout_guidance_loss(teacher_states, student_states)

            # Las cabezas reward/cont del alumno se omiten intencionalmente (Raw2Drive Seccion 3.3):
            # entrenarlas con sensores brutos causa divergencia porque los frames adyacentes son muy
            # similares mientras reward/cont fluctúan abruptamente. El Head Guidance usa las cabezas
            # estables del profesor como señal de supervisión. En inferencia, el alumno solo necesita
            # su propio actor (ya entrenado), las cabezas reward/cont no se usan en producción.
            wm_loss = recon_loss + kl_loss + guidance_loss

            # Borra gradientes antiguos, propaga la perdida hacia atras y actualiza los pesos
            opt_wm.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_wm.parameters(), max_norm=1000.0)  # DreamerV3: previene gradientes muy altos en secuencias largas
            opt_wm.step()

            # --- Actor-Critic por imaginación con Head Guidance ---
            # Las cabezas del profesor (head_guidance) estiman reward/cont sobre los estados del alumno
            start = flatten_states(student_states)      # [B*T, dim] desconectado
            actor_loss, critic_loss, policy_metrics = imagine_losses(
                student_wm.rssm, actor, critic, start, head_guidance.reward_fn, head_guidance.cont_fn, config)

            # Actor
            # Borra gradientes antiguos, propaga la perdida hacia atras y actualiza los pesos
            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1000.0)
            opt_actor.step()

            # Critic
            # Borra gradientes antiguos, propaga la perdida hacia atras y actualiza los pesos
            opt_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1000.0)
            opt_critic.step()

            if episode % 50 == 0:
                branch = config.get('branch', 'x')
                print(
                    f"[student-{branch} ep={episode}] "
                    f"wm={wm_loss.item():.3f} recon={recon_loss.item():.3f} "
                    f"kl={kl_loss.item():.3f} guid={guidance_loss.item():.3f} | "
                    f"actor={policy_metrics['actor_loss']:.3f} critic={policy_metrics['critic_loss']:.3f} "
                    f"entropy={policy_metrics['entropy']:.3f} return={policy_metrics['return']:.2f}"
                )

            # Checkpoint periódico: permite reanudar el entrenamiento si se interrumpe
            automatic_checkpoint_episodes = int(train_config.get("automatic_checkpoint_episodes", 500))
            if episode > 0 and episode % automatic_checkpoint_episodes == 0:
                branch = config.get('branch', 'x')
                torch.save({
                    "episode": episode,
                    "wm": student_wm.state_dict(),
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "opt_wm": opt_wm.state_dict(),
                    "opt_actor": opt_actor.state_dict(),
                    "opt_critic": opt_critic.state_dict(),
                    "config": config,
                }, f"checkpoints/student_{branch}_ep{episode}.pt")

        branch = config.get('branch', 'x')
        # Guarda pesos, optimizadores, episodio actual y config para poder reanudar el entrenamiento
        torch.save({
            "episode": episode,
            "wm": student_wm.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "opt_wm": opt_wm.state_dict(),
            "opt_actor": opt_actor.state_dict(),
            "opt_critic": opt_critic.state_dict(),
            "config": config,
        }, f"checkpoints/student_{branch}.pt")

    finally:
        env.close()
    return student_wm, actor
