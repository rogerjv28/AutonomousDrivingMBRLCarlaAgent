"""Trucos numéricos DreamerV3: symlog/symexp, two-hot, latente categórico (straight-through), KL balancing."""

import torch
import torch.nn.functional as F


def symlog(x):
    """Comprime magnitudes grandes preservando el signo (log), para que valores con escalas muy
    dispares (p.ej. -1 por colisión vs. +1000 de retorno acumulado) quepan en pocos bins.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    """Inversa exacta de symlog, para recuperar el valor real tras predecir en espacio symlog."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def make_bins(num_bins: int, min_value: float = -20.0, max_value: float = 20.0, device=None):
    return torch.linspace(min_value, max_value, num_bins, device=device)


def two_hot(value: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """value: [...]; devuelve [..., K] con la codificación two-hot sobre `bins` (ordenados).

    En vez de un one-hot exacto sobre el bin más cercano, reparte la masa entre los dos bins
    vecinos de `value` según su distancia a cada uno — así el valor real queda codificado sin
    discretizar del todo.
    """
    num_bins = bins.numel()
    value = value.clamp(float(bins[0]), float(bins[-1]))

    # Bucketize da el índice del primer bin >= value, se acota a [1, num_bins-1] para que
    # bin_index-1 y bin_index sean siempre válidos (value queda entre dos bins reales).
    bin_index = torch.bucketize(value, bins).clamp(1, num_bins - 1)
    lower_bin = bins[bin_index - 1]
    upper_bin = bins[bin_index]
    upper_weight = (value - lower_bin) / (upper_bin - lower_bin + 1e-8)

    two_hot_encoding = torch.zeros(*value.shape, num_bins, device=value.device, dtype=value.dtype)
    two_hot_encoding.scatter_(-1, (bin_index - 1).unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
    two_hot_encoding.scatter_(-1, bin_index.unsqueeze(-1), upper_weight.unsqueeze(-1))

    return two_hot_encoding


def from_probs(probs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return (probs * bins).sum(-1)


def two_hot_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy entre logits [...,K] y el target escalar (en espacio symlog) two-hot."""
    target_two_hot = two_hot(symlog(target), bins)
    log_probs = F.log_softmax(logits, dim=-1)
    
    return -(target_two_hot * log_probs).sum(-1)


def st_onehot_sample(logits: torch.Tensor) -> torch.Tensor:
    """Muestra one-hot con straight-through. logits: [..., C]."""
    dist = torch.distributions.OneHotCategorical(logits=logits)
    sample = dist.sample()
    # En valor da exactamente `sample` (sample + probs - probs = sample), pero al derivar
    # probs.detach() no aporta gradiente, así que el gradiente fluye como si la salida fuera probs.
    return sample + dist.probs - dist.probs.detach()


def categorical_kl_balance(post_logits, prior_logits, free_bits=1.0, balance=0.8):
    """KL entre posterior y prior categóricos [B, S, C], con balancing y free bits. Devuelve [B].

    KL balancing (DreamerV3): calcula el mismo KL dos veces cortando el gradiente hacia un lado
    cada vez, para ponderar por separado el ritmo al que se mueve el prior hacia el posterior y
    viceversa (si no, el optimizador tiende a colapsar el posterior hacia el prior).
    """

    def kl(logits_a, logits_b):
        dist_a = torch.distributions.Categorical(logits=logits_a)
        dist_b = torch.distributions.Categorical(logits=logits_b)
        return torch.distributions.kl_divergence(dist_a, dist_b).sum(-1)  # suma sobre S

    kl_prior = kl(post_logits.detach(), prior_logits)   # gradiente solo al prior
    kl_post = kl(post_logits, prior_logits.detach())    # gradiente solo al posterior
    kl_prior = torch.clamp(kl_prior, min=free_bits)     # free bits: evita el colapso del posterior
    kl_post = torch.clamp(kl_post, min=free_bits)

    return balance * kl_prior + (1.0 - balance) * kl_post
