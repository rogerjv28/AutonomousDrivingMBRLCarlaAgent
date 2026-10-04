"""Etapa 1: entrenamiento del World Model privilegiado y la política del profesor. Requiere CARLA.

`TeacherTrainer` completa los hooks de `BaseTrainer` con las cabezas de reward/cont
propias: el profesor entrena su world model completo (recon+reward+cont+kl) y usa sus propias
cabezas como reward_fn/cont_fn de la imaginación (es la fuente del guidance).
"""
import torch

from adcarla.carla_env.env import CarlaEnv
from adcarla.encoders.factory import build_encoder
from adcarla.policy.factory import build_actor, build_critic
from adcarla.training.agent import RolloutAgent
from adcarla.training.base_trainer import BaseTrainer
from adcarla.utils.distributions import from_probs, symexp
from adcarla.world_model.world_model import WorldModel


class TeacherTrainer(BaseTrainer):
    """Entrena el world model privilegiado y la política del profesor (Etapa 1)."""

    def _build_env(self):
        self.config = {**self.config, "encoder": "privileged", "privileged_bev": True}
        return CarlaEnv(self.config)

    def _build_models(self):
        wm = WorldModel(self.config, build_encoder(self.config), self.num_actions).to(self.device)
        actor = build_actor(self.config, wm.rssm.feat_dim, self.num_actions).to(self.device)
        critic = build_critic(self.config, wm.rssm.feat_dim).to(self.device)
        agent = RolloutAgent(wm, actor, self.config, self.device)
        return wm, actor, critic, agent

    def compute_wm_loss(self, batch: dict):
        """El profesor entrena los 4 términos: es la única fuente de guidance de los alumnos."""
        return self.wm.loss(batch)

    def get_reward_fn(self):
        def reward_fn(feat):
            return symexp(from_probs(torch.softmax(self.wm.reward(feat), -1), self.wm.bins))
        return reward_fn

    def get_cont_fn(self):
        def cont_fn(feat):
            return torch.sigmoid(self.wm.cont(feat))
        return cont_fn


def train_teacher(config: dict, resume: bool = False):
    """Punto de entrada de scripts/train_teacher.py.

    Args:
        config: configuración completa (ver configs/teacher.yaml).
        resume: si es True, reanuda desde el último checkpoint guardado.

    Returns:
        Tupla (wm, actor, critic) con los módulos entrenados.
    """
    trainer = TeacherTrainer(config)
    trainer.train(resume=resume)
    return trainer.wm, trainer.actor, trainer.critic
