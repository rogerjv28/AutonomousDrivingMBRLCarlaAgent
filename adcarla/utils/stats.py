"""Estadística del protocolo de evaluación: media ± IC 95 % y Mann-Whitney U entre ramas.

Racional de la unidad estadística (semilla, no episodio) y del suelo de p alcanzable con pocas semillas.
"""
import warnings
from math import factorial, sqrt

from scipy import stats as scipy_stats

CONFIDENCE = 0.95


def mean_ci95(values, confidence: float = CONFIDENCE) -> dict:
    """Media e intervalo de confianza con la t de Student (muestras pequeñas).

    Args:
        values: valores de la métrica, uno por semilla.
        confidence: nivel de confianza; 0.95 por defecto.

    Returns:
        Dict con "mean", "ci95" (semiancho del intervalo; None si n < 2), "low", "high" y "n".
    """
    sample = [float(v) for v in values]
    n = len(sample)
    if n == 0:
        raise ValueError("mean_ci95 necesita al menos un valor")

    mean = sum(sample) / n
    if n < 2:
        # Con una sola semilla no hay varianza que estimar. Devolver 0 sugeriría certeza absoluta.
        return {"mean": mean, "ci95": None, "low": None, "high": None, "n": n}

    variance = sum((v - mean) ** 2 for v in sample) / (n - 1)
    standard_error = sqrt(variance) / sqrt(n)
    t = float(scipy_stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    margin_of_error = t * standard_error

    return {"mean": mean, "ci95": margin_of_error, "low": mean - margin_of_error,
            "high": mean + margin_of_error, "n": n}


def min_p_value(n_a: int, n_b: int) -> float:
    """p bilateral más pequeño que puede dar Mann-Whitney exacto con esos tamaños muestrales.

    Es `2 / C(n_a + n_b, n_a)`: solo una de las combinaciones posibles deja las dos muestras
    perfectamente separadas (y otra la separación inversa). Con 3 vs 3 sale 0.1, así que ningún
    resultado del TFM con 3 semillas puede alcanzar p < 0.05 por muy separadas que estén las ramas.
    """
    total = int(n_a) + int(n_b)
    combinations = factorial(total) // (factorial(int(n_a)) * factorial(int(n_b)))
    return 2.0 / combinations


def mann_whitney(a, b) -> dict:
    """Test de Mann-Whitney U bilateral entre dos ramas (no paramétrico).

    Args:
        a, b: valores de la métrica por semilla en cada rama (al menos 2 por rama).

    Con empates, usa `method="exact"` incluso cuando scipy elegiría la aproximación normal:
    es conservador, nunca infla la significancia.

    Returns:
        Dict con "u", "p_value", "n_a", "n_b" y "min_p_value" (el suelo del test, ver arriba).

    Raises:
        ValueError: si alguna rama trae menos de dos semillas.
    """
    sample_a = [float(v) for v in a]
    sample_b = [float(v) for v in b]
    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError(
            f"Mann-Whitney necesita >= 2 semillas por rama y llegaron {len(sample_a)} y "
            f"{len(sample_b)}; sube eval.seeds o --seeds.")

    floor_p_value = min_p_value(len(sample_a), len(sample_b))

    # Muestras idénticas: scipy no puede tipificar (la corrección por empates anula la varianza)
    # y devolvería NaN. Sin ninguna diferencia observada, el p correcto es 1.
    if sample_a == sample_b or len(set(sample_a + sample_b)) == 1:
        u = len(sample_a) * len(sample_b) / 2.0
        return {"u": u, "p_value": 1.0, "n_a": len(sample_a), "n_b": len(sample_b),
                "min_p_value": floor_p_value}

    # Método exacto con muestras pequeñas, aunque haya empates (ver docstring de mann_whitney).
    method = "exact" if len(sample_a) <= 8 and len(sample_b) <= 8 else "auto"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # scipy avisa de los empates; es la elección de arriba
        result = scipy_stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided",
                                          method=method)
    return {"u": float(result.statistic), "p_value": float(result.pvalue),
            "n_a": len(sample_a), "n_b": len(sample_b), "min_p_value": floor_p_value}
