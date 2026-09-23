"""WorldModel = encoder + RSSM + heads. loss(batch, terms=...) = recon(BEV) + reward(two-hot) +
continue + KL(dyn/rep), con los coeficientes beta_pred/beta_dyn/beta_rep de DreamerV3 Ec. 2.

`_encode_seq` trocea la secuencia en chunks (`train.encode_chunk_size`) y aplica ahí
`train.grad_checkpoint`/`train.amp`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .rssm import RSSM
from .heads import Decoder, RewardHead, ContinueHead
from adcarla.utils.distributions import make_bins, two_hot_loss, categorical_kl_balance


class WorldModel(nn.Module):
    """Modelo del mundo completo: encoder de sensores + RSSM + cabezas de predicción.

    Aprende a modelar el entorno de conducción en un espacio latente compacto. `loss()` combina
    hasta cuatro términos (recon, reward, cont, kl; ver `TERMS`), seleccionables con `terms=` según
    detalla su propio docstring.
    """

    TERMS = ("recon", "reward", "cont", "kl")   # términos disponibles para `loss(batch, terms=...)`
    ENCODE_CHUNK_SIZE = 64   # valor por defecto si train.encode_chunk_size no esta en el YAML

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

        self.free_bits = float(wm_config["free_bits"])  # suelo de KL total por timestep (evita colapso)
        # Coeficientes de la Ec. 2 de DreamerV3: pred pondera recon+reward+cont
        # beta_pred: si es bajo, el world model ignora los sensores. Si es alto, aprende muy precisamente dónde está.
        # beta_dyn/beta_rep modelan el equilibrio del KL
        # beta_dyn: fuerza el prior a aprender del historial (h) sin depender de sensores
        # beta_rep: fuerza el posterior a aprender de los sensores (embed) sin ignorar el prior
        self.beta_pred = float(wm_config["beta_pred"])
        self.beta_dyn = float(wm_config["beta_dyn"])
        self.beta_rep = float(wm_config["beta_rep"])
        self.register_buffer("bins", make_bins(int(wm_config["num_bins"]),
                                                float(wm_config["bin_min"]),
                                                float(wm_config["bin_max"])))  # bins two-hot fijos (D6)

        train_config = config.get("train", {})
        self.grad_checkpoint = bool(train_config.get("grad_checkpoint", False))
        self.amp = bool(train_config.get("amp", False))
        # Válvula si el batch no cabe en la GPU alquilada (docs/implementation_decisions.md,
        # sección World model).
        self.encode_chunk_size = int(train_config.get("encode_chunk_size", self.ENCODE_CHUNK_SIZE))
        self.amp_device_type = "cuda" if str(config.get("device", "cpu")).startswith("cuda") else "cpu"

    def _encode_flat(self, keys: tuple, *tensors: torch.Tensor):
        """Encoder sobre un chunk ya aplanado [K, ...]. Recibe los tensores sueltos (no un dict)
        para que `torch.utils.checkpoint` los reconozca como entradas y pueda recomputar este
        forward en el backward sin guardar sus activaciones intermedias.

        Returns:
            Tupla (embedding [K, embed_dim], rejilla BEV [K, grid_channels, G, G]).
        """
        return self.encoder.forward_with_grid(dict(zip(keys, tensors)))

    def _encode_seq(self, batch: dict, with_grid: bool = False):
        """Aplica el encoder a la secuencia en chunks de como mucho `encode_chunk_size` timesteps.

        Trocear es una simple re-partición (cada timestep es independiente de los demás para el
        encoder): el resultado es idéntico, solo cambia cuántas activaciones hay vivas a la vez.

        Args:
            batch: dict de tensores [B, T, ...], el encoder lee sus propias claves (cameras, lidar_bev...).
            with_grid: si es True devuelve también la rejilla BEV de características, que es lo que
                alinea el rollout guidance celda a celda (Ec. 1 de Raw2Drive). Sale del mismo
                forward que el embedding, así que no cuesta un pase extra del encoder; solo se
                pide cuando hace falta porque mantener [B, T, C, G, G] viva ocupa memoria.

        Returns:
            Embeddings [B, T, embed_dim], o la tupla (embeddings, rejillas [B, T, C, G, G])
            si `with_grid`.
        """
        B, T = batch["prev_action"].shape[:2]
        # Filtra por las claves que el encoder de esta rama declara (BEVEncoder.input_keys): p.ej.
        # el profesor lee "bev" (~58 MB por trozo) pero ningún encoder de alumno lo hace, y sin
        # este filtro viajaba igual a través de cada chunk, deshaciendo parte del ahorro de
        # memoria del troceado.
        tensor_items = {k: v for k, v in batch.items()
                        if k in self.encoder.input_keys and torch.is_tensor(v) and v.dim() >= 2}
        keys = tuple(tensor_items.keys())

        embeds, grids = [], []
        for start in range(0, T, self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, T)
            chunk_len = end - start
            tensors = tuple(v[:, start:end].reshape(B * chunk_len, *v.shape[2:]) for v in tensor_items.values())

            if self.grad_checkpoint and torch.is_grad_enabled():
                embed_chunk, grid_chunk = checkpoint(self._encode_flat, keys, *tensors, use_reentrant=False)
            else:
                embed_chunk, grid_chunk = self._encode_flat(keys, *tensors)
            embeds.append(embed_chunk.reshape(B, chunk_len, -1))
            if with_grid:
                grids.append(grid_chunk.reshape(B, chunk_len, *grid_chunk.shape[1:]))

        embed = torch.cat(embeds, dim=1)     # [B, T, embed_dim]

        return (embed, torch.cat(grids, dim=1)) if with_grid else embed

    def loss(self, batch: dict, terms: tuple = TERMS, aux: dict = None):
        """Calcula la pérdida del world model sobre un batch de secuencias (DreamerV3 Ec. 2).

        `terms` deja elegir qué términos calcular: el alumno solo necesita `("recon", "kl")`
        porque sus cabezas reward/cont no se entrenan. Así no se llama ni se hace forward de
        las cabezas que no le tocan, en vez de calcularlas y descartarlas.

        Args:
            batch: dict con tensores [B, T, ...]:
                - "prev_action": acción previa (a_{t-1}) ya en one-hot [B, T, num_actions], el
                  primer paso de cada episodio lleva el vector nulo (fix P1-1: el posterior de
                  o_t debe condicionarse en la acción que LLEVÓ a o_t, no en la que se toma en o_t).
                - "bev": máscaras BEV privilegiadas [B, T, bev_channels, size, size].
                - "reward": recompensas escalares [B, T].
                - "cont": flags de continuación del episodio [B, T], 1=continúa/0=done.
                - claves del encoder (cameras, lidar_bev...).
            terms: subconjunto de `WorldModel.TERMS` a calcular.
            aux: dict opcional; si se pasa, se rellena con "bev_grid" [B, T, C, G, G] y
                "post_logits" [B, T, stoch_dim, stoch_classes] del mismo forward, que es lo que
                necesita el rollout guidance del alumno.

        Returns:
            Tupla (loss escalar, states dict [B,T,...] en fp32, metrics dict con un valor por
            término calculado más "loss").
        """
        B, T = batch["prev_action"].shape[:2]

        # AMP: todo el forward corre en bf16 si train.amp=true. PyTorch promueve
        # automáticamente a fp32 las operaciones que lo necesitan (BCE, softmax...) dentro del
        # propio autocast, así que las pérdidas se calculan aquí dentro sin trato especial.
        with torch.autocast(device_type=self.amp_device_type, dtype=torch.bfloat16, enabled=self.amp):
            if aux is None:
                embed = self._encode_seq(batch) # [B, T, embed_dim]
            else:
                embed, aux["bev_grid"] = self._encode_seq(batch, with_grid=True)
            states, post, prior, _ = self.rssm.observe(embed, batch["prev_action"]) # estados latentes
            if aux is not None:
                aux["post_logits"] = post
            feat = self.rssm.feat(states)       # [B, T, feat_dim]

            total = torch.zeros(B, T, device=feat.device)
            metrics = {}

            if "recon" in terms:
                feat_flat = feat.reshape(B * T, -1)
                recon = self.decoder(feat_flat).reshape(B, T, self.bev_channels, self.size, self.size)
                # Log-verosimilitud (DreamerV3 Ec. 2): suma sobre [C, H, W].
                recon_loss = F.binary_cross_entropy_with_logits(recon, batch["bev"], reduction="none").sum([2, 3, 4])
                total = total + self.beta_pred * recon_loss
                metrics["recon"] = recon_loss.mean().item()

            if "reward" in terms:
                reward_loss = two_hot_loss(self.reward(feat), batch["reward"], self.bins)
                total = total + self.beta_pred * reward_loss
                metrics["reward"] = reward_loss.mean().item()

            if "cont" in terms:
                cont_loss = F.binary_cross_entropy_with_logits(self.cont(feat), batch["cont"], reduction="none")
                total = total + self.beta_pred * cont_loss
                metrics["cont"] = cont_loss.mean().item()

            if "kl" in terms:
                # KL entre posterior (con observación) y prior (sin observación): regula la calidad del prior
                kl = categorical_kl_balance(post, prior, self.free_bits, self.beta_dyn, self.beta_rep)
                total = total + kl
                metrics["kl"] = kl.mean().item()

            loss = total.mean()
            metrics["loss"] = loss.item()

        # RSSM corre en bf16 dentro del autocast (si amp=true), pero actor/critic y guidance
        # necesitan fp32 (sus Linear/GRUCell fallan con bf16 inputs + fp32 weights).
        # Conversión explícita aquí para garantizar coherencia de tipos en el siguiente stage.
        # Si amp=false, .float() es un no-op (ya están en fp32).
        states = {k: v.float() for k, v in states.items()}
        for clave in (aux or {}):
            aux[clave] = aux[clave].float()   # fp32 fuera del autocast, mismo motivo que `states`
        return loss, states, metrics
