"""Guidance (Raw2Drive Seccion 3.3):
- RolloutGuidance: alinea los estados latentes (h, stoch) del profesor y el alumno con MSE.
- HeadGuidance: usa las heads de reward/continue del profesor para guiar la política del alumno.
"""
import torch
import torch.nn.functional as F
from adcarla.utils.distributions import from_probs, symexp


def rollout_guidance_loss(teacher_states: dict, student_states: dict) -> torch.Tensor:
    """MSE entre el feat (h,stoch) del profesor y el del alumno, por timestep."""
    teacher_states_concat = torch.cat([teacher_states["h"], teacher_states["stoch"]], -1).detach()
    student_states_concat = torch.cat([student_states["h"], student_states["stoch"]], -1)
    return F.mse_loss(student_states_concat, teacher_states_concat)


class HeadGuidance:
    """Envuelve las heads del profesor para dar reward_fn/cont_fn sobre el feat compartido con el alumno."""
    def __init__(self, teacher_wm):
        self.reward_head = teacher_wm.reward
        self.cont_head = teacher_wm.cont
        self.bins = teacher_wm.bins

    def reward_fn(self, feat: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.reward_head(feat), dim=-1)
        return symexp(from_probs(probs, self.bins))

    def cont_fn(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cont_head(feat))
