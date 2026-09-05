"""Guidance profesor-alumno (Raw2Drive Sec. 3.3): `RolloutGuidance` alinea rejilla BEV, estado
estocástico y determinista del alumno con los del profesor. `HeadGuidance` hace rodar al
profesor en paralelo dentro de la imaginación para aportar reward/cont y destilar su política.
`teacher_probability` calcula la fracción de episodios con el profesor al volante durante
la recolección mixta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from adcarla.utils.distributions import from_probs, symexp, unimix_logits


class RolloutGuidance(nn.Module):
    """Ec. 1 de Raw2Drive: alinea los tres estados del rollout del alumno con los del profesor.

    L = beta_e * sum_celdas MSE(e, e^) + beta_s * KL(s || s^) + beta_h * MSE(h, h^)

    El profesor es el objetivo y va siempre con `detach()`: la pérdida solo mueve al alumno.
    Requiere que el profesor haya recorrido la secuencia con `stoch_override` = la muestra del
    alumno (ver `RSSM.obs_step`). Si cada stream muestrea la suya, los tres términos miden sobre
    todo ruido de muestreo y el guidance nunca baja de ese suelo.
    """

    def __init__(self, config: dict, student_grid_channels: int, teacher_grid_channels: int):
        """
        Args:
            config: configuración completa; lee el bloque `guidance:` (beta_encoder/stoch/deter).
            student_grid_channels: canales de la rejilla `bev_features()` del alumno.
            teacher_grid_channels: canales de la rejilla `bev_features()` del profesor.
        """
        super().__init__()
        guidance_config = config.get("guidance", {}) or {}
        self.beta_encoder = float(guidance_config.get("beta_encoder", 10.0))
        self.beta_stoch = float(guidance_config.get("beta_stoch", 10.0))
        self.beta_deter = float(guidance_config.get("beta_deter", 5.0))

        # Con los configs reales las dos rejillas tienen los mismos canales (128) y no hay
        # proyección: el guidance compara directamente celda a celda.
        self.projection = None
        if student_grid_channels != teacher_grid_channels:
            self.projection = nn.Conv2d(student_grid_channels, teacher_grid_channels, kernel_size=1)

    def forward(self, student: dict, teacher: dict):
        """Calcula la pérdida de rollout guidance de un batch de secuencias.

        Args:
            student: dict del alumno con "grid" [B,T,C,G,G], "post_logits" [B,T,S,C] y "h" [B,T,D].
            teacher: el mismo dict del profesor (se usa siempre desconectado del grafo).

        Returns:
            Tupla (pérdida escalar, dict de métricas con los tres términos ya ponderados).
        """
        student_grid = student["grid"]
        if self.projection is not None:
            B, T = student_grid.shape[:2]
            student_grid = self.projection(student_grid.flatten(0, 1)).reshape(B, T, *teacher["grid"].shape[2:])

        # Spatial-Temporal Alignment: MSE de cada celda (media sobre canales) sumado sobre las
        # grid_num celdas. La reducción sobre [B, T] es media (una suma sobre t solo cambia la
        # escala, que absorbe el learning rate).
        encoder_term = ((student_grid - teacher["grid"].detach()) ** 2).mean(2).sum([2, 3]).mean()

        # Abstract-State Alignment. KL(profesor || alumno): el profesor es el objetivo, como en la
        # Ec. 1 (KL(s_t, s^_t), s_t = stream privilegiado). Suma sobre las variables categóricas.
        teacher_dist = torch.distributions.Categorical(logits=unimix_logits(teacher["post_logits"].detach()))
        student_dist = torch.distributions.Categorical(logits=unimix_logits(student["post_logits"]))
        stoch_term = torch.distributions.kl_divergence(teacher_dist, student_dist).sum(-1).mean()

        deter_term = F.mse_loss(student["h"], teacher["h"].detach())

        encoder_term = self.beta_encoder * encoder_term
        stoch_term = self.beta_stoch * stoch_term
        deter_term = self.beta_deter * deter_term
        loss = encoder_term + stoch_term + deter_term

        metrics = {"guidance": loss.item(),
                   "guidance_encoder": encoder_term.item(),
                   "guidance_stoch": stoch_term.item(),
                   "guidance_deter": deter_term.item()}
        return loss, metrics


class HeadGuidance:
    """Head Guidance por pseudo-deducción (Raw2Drive Sec. 3.3 y Fig. 5.

    El world model del profesor rueda en paralelo al del alumno durante la imaginación y sus
    cabezas dan reward/cont sobre el feat del profesor. Orden de uso: `reset()` con el estado del
    profesor del mismo instante en que arranca la imaginación, `step()` una vez por paso (lo llama
    `RSSM.imagine` vía `on_step`) y `feats()`/`distill_loss()` al final.
    """

    def __init__(self, config: dict, teacher_wm, teacher_actor):
        """
        Args:
            config: configuración completa; lee `guidance.distill_weight`.
            teacher_wm: world model privilegiado del profesor, congelado.
            teacher_actor: actor del profesor, congelado (objetivo de la destilación).
        """
        self.teacher_wm = teacher_wm
        self.teacher_actor = teacher_actor
        self.distill_weight = float((config.get("guidance", {}) or {})["distill_weight"])
        self._state = None
        self._feats = []

    def reset(self, teacher_state: dict):
        """Fija el estado del profesor del que arranca el rollout paralelo ([K, dim], detached)."""
        self._state = {k: v.detach() for k, v in teacher_state.items()}
        self._feats = []

    @torch.no_grad()
    def step(self, action: torch.Tensor, stoch: torch.Tensor):
        """Avanza el profesor un paso con la acción y la muestra estocástica del alumno.

        Args:
            action: [K, num_actions] acción one-hot que acaba de tomar el actor del alumno.
            stoch: [K, stoch_flat] muestra estocástica del alumno en ese paso.
        """
        if self._state is None:
            raise RuntimeError("HeadGuidance.step() antes de reset(): falta el estado del profesor")
        self._state, _ = self.teacher_wm.rssm.img_step(
            self._state, action.detach(), stoch_override=stoch.detach())
        self._feats.append(self.teacher_wm.rssm.feat(self._state))

    def feats(self) -> torch.Tensor:
        """Feats del profesor a lo largo del rollout imaginado [K, horizon, feat_dim]."""
        if not self._feats:
            raise RuntimeError("HeadGuidance.feats() sin pasos: falta rodar el rollout del profesor")
        return torch.stack(self._feats, dim=1)

    @torch.no_grad()
    def reward_fn(self, feat: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.teacher_wm.reward(feat), dim=-1)
        return symexp(from_probs(probs, self.teacher_wm.bins))

    @torch.no_grad()
    def cont_fn(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.teacher_wm.cont(feat))

    def distill_loss(self, student_logits: torch.Tensor) -> torch.Tensor:
        """Destilación de acciones: `distill_weight · CE(alumno, sg(profesor))`.

        Args:
            student_logits: [K, horizon, num_actions] logits del actor del alumno sobre SU feat.

        Returns:
            Escalar ya ponderado. Con los demás términos del actor a cero, minimizarlo solo es
            imitation learning del experto (Plan B del TFM).
        """
        with torch.no_grad():
            teacher_probs = torch.softmax(self.teacher_actor.logits(self.feats()), dim=-1)
        cross_entropy = -(teacher_probs * F.log_softmax(student_logits, dim=-1)).sum(-1).mean()
        return self.distill_weight * cross_entropy


def teacher_probability(episode: int, total_episodes: int, fraction: float) -> float:
    """Probabilidad de que el profesor conduzca el episodio: decae linealmente de 1 a 0
    en la primera `fraction` del entrenamiento.

    Args:
        episode: episodio actual.
        total_episodes: episodios totales del entrenamiento (`train.total_episodes`).
        fraction: fracción del entrenamiento en la que se agota el decay.

    Returns:
        p_teacher en [0, 1].
    """
    decay_episodes = fraction * total_episodes
    if decay_episodes <= 0:
        return 0.0
    return float(max(0.0, 1.0 - episode / decay_episodes))
