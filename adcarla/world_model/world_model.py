"""WorldModel = encoder + RSSM + heads. loss = recon(BEV) + reward(two-hot) + continue + KL(balanced)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rssm import RSSM
from .heads import Decoder, RewardHead, ContinueHead
from adcarla.utils.distributions import make_bins, two_hot_loss, categorical_kl_balance


class WorldModel(nn.Module):
    """Modelo del mundo completo: encoder de sensores + RSSM + cabezas de predicción.

    Aprende a modelar el entorno de conducción en un espacio latente compacto.
    El entrenamiento minimiza cuatro términos de pérdida:
    - recon: reconstrucción del mapa BEV privilegiado desde el estado latente.
    - reward: predicción de la recompensa (two-hot loss).
    - cont: predicción de si el episodio continúa (binary cross-entropy).
    - KL: regularización que acerca el prior al posterior (balanceada).
    """

    def __init__(self, config: dict, encoder: nn.Module, num_actions: int):
        """Construye el world model a partir de la configuración y el encoder de sensores.

        Args:
            config: configuración completa del experimento (ver configs/base.yaml).
            encoder: encoder de sensores ya instanciado (rama visión o fusión).
            num_actions: número de acciones discretas del espacio de acción.
        """
        super().__init__()
        wm_config = config["world_model"]
        self.num_actions = num_actions
        self.size = int(config["bev"]["size"])
        self.bev_channels = int(config["bev"]["channels"])

        self.encoder = encoder
        self.rssm = RSSM(
            action_dim=num_actions,
            embed_dim=int(wm_config["embed_dim"]),
            deter_dim=int(wm_config["deterministic_dim"]),
            stoch_dim=int(wm_config["stochastic_dim"]),
            stoch_classes=int(wm_config["stochastic_classes"]),
            hidden_dim=int(wm_config["hidden_dim"]),
        )

        feat_dim = self.rssm.feat_dim   # deter_dim + stoch_dim * stoch_classes

        # Cabezas de predicción que supervisan el estado latente durante el entrenamiento
        self.decoder = Decoder(feat_dim, self.bev_channels, self.size)
        self.reward = RewardHead(feat_dim, int(wm_config["num_bins"]))
        self.cont = ContinueHead(feat_dim)

        self.free_bits = float(wm_config["free_bits"])  # mínimo de KL por timestep (evita colapso)
        self.kl_balance = float(wm_config["kl_balance"])    # balance prior/posterior en la pérdida KL
        self.register_buffer("bins", make_bins(int(wm_config["num_bins"]))) # bins two-hot fijos

    def _encode_seq(self, batch: dict) -> torch.Tensor:
        """Aplica el encoder a cada timestep de la secuencia y devuelve los embeddings.

        Aplana la dimensión temporal (B,T → B*T) para procesar todos los timesteps
        en una sola llamada al encoder, luego restaura la dimensión temporal.

        Args:
            batch: dict de tensores [B, T, ...], el encoder lee sus propias claves (cameras, lidar_bev...).

        Returns:
            Embeddings [B, T, embed_dim].
        """
        B, T = batch["action"].shape[:2]
        # Aplana B y T para pasar todos los timesteps de golpe al encoder
        flat_batch = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.dim() >= 2:
                flat_batch[k] = v.reshape(B * T, *v.shape[2:])

        embed = self.encoder(flat_batch)    # [B*T, embed_dim]
        return embed.reshape(B, T, -1)      # [B, T, embed_dim]

    def loss(self, batch: dict):
        """Calcula la pérdida total del world model sobre un batch de secuencias.

        Args:
            batch: dict con tensores [B, T, ...]:
                - "action": acciones discretas [B, T].
                - "bev": máscaras BEV privilegiadas [B, T, bev_channels, size, size].
                - "reward": recompensas escalares [B, T].
                - "cont": flags de continuación del episodio [B, T] (1=continúa, 0=done).
                - claves del encoder (cameras, lidar_bev...).

        Returns:
            Tupla (loss escalar, states dict [B,T,...], metrics dict).
        """
        B, T = batch["action"].shape[:2]

        embed = self._encode_seq(batch)                                             # [B, T, embed_dim]
        action_oh = F.one_hot(batch["action"].long(), self.num_actions).float()     # [B, T, num_actions]
        states, post, prior, _ = self.rssm.observe(embed, action_oh)                # estados latentes
        feat = self.rssm.feat(states)                                               # [B, T, feat_dim]
        feat_flat = feat.reshape(B * T, -1)                                         # [B*T, feat_dim]

        # Reconstrucción del BEV: el decoder tiene que recuperar la máscara BEV desde el estado latente
        recon = self.decoder(feat_flat).reshape(B, T, self.bev_channels, self.size, self.size)
        recon_loss = F.binary_cross_entropy(recon, batch["bev"], reduction="none").mean([2, 3, 4])

        # Predicción de recompensa y continuación del episodio
        reward_loss = two_hot_loss(self.reward(feat), batch["reward"], self.bins)
        cont_loss = F.binary_cross_entropy_with_logits(self.cont(feat), batch["cont"], reduction="none")

        # KL entre posterior (con observación) y prior (sin observación): regula la calidad del prior
        kl = categorical_kl_balance(post, prior, self.free_bits, self.kl_balance)   # [B, T]

        loss = (recon_loss + reward_loss + cont_loss + kl).mean()
        metrics = {
            "recon": recon_loss.mean().item(),
            "reward": reward_loss.mean().item(),
            "cont": cont_loss.mean().item(),
            "kl": kl.mean().item(),
            "loss": loss.item(),
        }
        
        return loss, states, metrics
