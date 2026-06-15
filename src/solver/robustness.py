"""
robustness.py  (src/solver/robustness.py)
==========================================
Análisis de robustez estocástica de una ruta de aprendizaje mediante
simulación Monte Carlo sobre la incertidumbre en la duración de los cursos.

Motivación:
    Las duraciones d(v) del dataset son estimaciones nominales. Un estudiante
    real puede tardar más o menos. Este módulo modela d(v) como variable
    aleatoria d'(v) ~ Gamma(μ=d(v), cv) y estima, mediante M simulaciones,
    la probabilidad de que la ruta S* siga siendo factible bajo las duraciones
    reales: P(Σ d'(v) ≤ T_max).

    Esto permite al sistema dar al usuario un mensaje como:
        "Tu ruta de 12 cursos tiene un 87% de probabilidad de completarse
         dentro de las 300 horas presupuestadas, asumiendo una variabilidad
         típica del ±20% en la duración de cada curso."

Modelo estocástico:
    Se usa la distribución Gamma porque:
      1. Está definida en ℝ⁺ — la duración nunca es negativa.
      2. Permite modelar asimetría positiva: es más probable que un curso
         tome MÁS tiempo del estimado que menos.
      3. Se parametriza naturalmente con μ (media) y cv (coef. de variación).

    Parametrización:
        shape (α) = (1/cv)²
        scale (θ) = μ · cv²
        ⟹ E[d'(v)] = μ = d(v)     (insesgado)
        ⟹ std[d'(v)] = μ · cv

    Valores típicos de cv:
        0.10 → variabilidad baja    (±10%):  cursos muy estructurados
        0.20 → variabilidad media   (±20%):  recomendado por defecto
        0.30 → variabilidad alta    (±30%):  cursos abiertos / proyectos
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado del análisis de robustez
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    """
    Resultado del análisis de robustez de una ruta de aprendizaje.

    Attributes:
        selected_ids:       IDs de los cursos en la ruta analizada.
        t_max:              Restricción de tiempo (horas).
        n_simulations:      Número de simulaciones Monte Carlo ejecutadas.
        cv:                 Coeficiente de variación usado en el modelo Gamma.
        p_feasible:         P(Σ d'(v) ≤ T_max) — probabilidad de factibilidad.
        mean_duration:      Duración media observada en las simulaciones.
        std_duration:       Desviación estándar de la duración simulada.
        ci_95_lower:        Límite inferior del IC 95% para p_feasible.
        ci_95_upper:        Límite superior del IC 95% para p_feasible.
        percentile_95_dur:  Percentil 95 de la duración simulada (h).
    """
    selected_ids: List[str]
    t_max: float
    n_simulations: int
    cv: float
    p_feasible: float
    mean_duration: float
    std_duration: float
    ci_95_lower: float
    ci_95_upper: float
    percentile_95_dur: float

    @property
    def risk_level(self) -> str:
        """Clasificación cualitativa del riesgo de incumplir T_max."""
        if self.p_feasible >= 0.90:
            return "BAJO  (p ≥ 90%)"
        elif self.p_feasible >= 0.70:
            return "MEDIO (70% ≤ p < 90%)"
        else:
            return "ALTO  (p < 70%)"

    def summary(self) -> str:
        return (
            f"Análisis de robustez — {len(self.selected_ids)} cursos\n"
            f"  T_max              : {self.t_max:.0f} h\n"
            f"  Coef. variación    : cv = {self.cv:.2f} (±{self.cv*100:.0f}%)\n"
            f"  Simulaciones       : M = {self.n_simulations:,}\n"
            f"  P(factible)        : {self.p_feasible:.1%}\n"
            f"  IC 95%             : [{self.ci_95_lower:.1%}, {self.ci_95_upper:.1%}]\n"
            f"  Duración media sim : {self.mean_duration:.1f} ± {self.std_duration:.1f} h\n"
            f"  Percentil 95 dur.  : {self.percentile_95_dur:.1f} h\n"
            f"  Nivel de riesgo    : {self.risk_level}"
        )


# ---------------------------------------------------------------------------
# Muestreador Gamma (solo stdlib, sin numpy/scipy)
# ---------------------------------------------------------------------------

def _gamma_sample(mu: float, cv: float, rng: random.Random) -> float:
    """
    Muestrea d'(v) ~ Gamma(μ=mu, cv=cv) usando el método de Marsaglia-Tsang
    implementado en random.gammavariate de la stdlib.

    Args:
        mu:  Media de la distribución (duración nominal del curso en horas).
        cv:  Coeficiente de variación (σ/μ). Default recomendado: 0.20.
        rng: Instancia de random.Random para reproducibilidad por hilo.

    Returns:
        Duración simulada en horas (siempre > 0).
    """
    if cv <= 0:
        return mu  # Sin variabilidad → retornar la media exacta

    # Parametrización Gamma: α = (1/cv)², β = μ·cv²
    alpha = (1.0 / cv) ** 2
    beta = mu * cv ** 2  # = μ/α → scale = μ·cv²

    return rng.gammavariate(alpha, beta)


# ---------------------------------------------------------------------------
# Análisis de robustez principal
# ---------------------------------------------------------------------------

def robustness_analysis(
    problem,                        # LearningPathProblem
    selected_ids: Sequence[str],
    n_simulations: int = 5000,
    cv: float = 0.20,
    seed: Optional[int] = 42,
) -> RobustnessResult:
    """
    Evalúa la robustez de una ruta S* bajo incertidumbre en las duraciones.

    Para cada simulación j = 1..M:
      1. Muestrear d'(v) ~ Gamma(μ=d(v), cv) para cada v ∈ S*.
      2. Computar duración total simulada: D_j = Σ d'(v).
      3. Registrar si D_j ≤ T_max (factible = 1, incumple = 0).
    Estimar:
      p_factible ≈ (1/M) Σ 1{D_j ≤ T_max}   ← estimador de Monte Carlo
      IC 95% via intervalo normal de proporciones (Wilson score).

    Args:
        problem:       Instancia del problema (para T_max y acceso a duraciones).
        selected_ids:  Lista de IDs de la ruta a evaluar (S*).
        n_simulations: Número de simulaciones M (mayor → IC más estrecho).
        cv:            Coeficiente de variación de las duraciones (default 0.20).
        seed:          Semilla para reproducibilidad.

    Returns:
        RobustnessResult con todas las métricas del análisis.

    Raises:
        ValueError: Si selected_ids contiene IDs no presentes en el problema.
    """
    rng = random.Random(seed)
    index = {c.id: c for c in problem.courses}

    # Validar que todos los IDs existen
    missing = [cid for cid in selected_ids if cid not in index]
    if missing:
        raise ValueError(f"IDs no encontrados en el problema: {missing}")

    t_max = problem.t_max
    courses_in_path = [index[cid] for cid in selected_ids]
    durations_nominal = [c.duration for c in courses_in_path]

    logger.info(
        "Robustez MC: instancia=%s | |S*|=%d | M=%d | cv=%.2f",
        problem.instance_id, len(selected_ids), n_simulations, cv,
    )

    # Bucle de simulación
    feasible_count = 0
    simulated_durations: List[float] = []

    for _ in range(n_simulations):
        # Muestrear duración real de cada curso en la ruta
        total = sum(
            _gamma_sample(d_nominal, cv, rng)
            for d_nominal in durations_nominal
        )
        simulated_durations.append(total)
        if total <= t_max:
            feasible_count += 1

    # Estadísticas de la distribución simulada
    mean_dur = sum(simulated_durations) / n_simulations
    variance = sum((d - mean_dur) ** 2 for d in simulated_durations) / (n_simulations - 1)
    std_dur = math.sqrt(variance)

    # Percentil 95 (sin numpy)
    sorted_durs = sorted(simulated_durations)
    p95_idx = int(0.95 * n_simulations)
    p95_dur = sorted_durs[min(p95_idx, n_simulations - 1)]

    # Estimador MC de P(factible)
    p_hat = feasible_count / n_simulations

    # Intervalo de confianza 95% (Wilson score interval — más robusto que normal)
    z = 1.96
    denom = 1 + z ** 2 / n_simulations
    center = (p_hat + z ** 2 / (2 * n_simulations)) / denom
    half_width = (z / denom) * math.sqrt(
        p_hat * (1 - p_hat) / n_simulations + z ** 2 / (4 * n_simulations ** 2)
    )
    ci_lower = max(0.0, center - half_width)
    ci_upper = min(1.0, center + half_width)

    result = RobustnessResult(
        selected_ids=list(selected_ids),
        t_max=t_max,
        n_simulations=n_simulations,
        cv=cv,
        p_feasible=p_hat,
        mean_duration=mean_dur,
        std_duration=std_dur,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        percentile_95_dur=p95_dur,
    )

    # CORRECCIÓN: la llamada original usaba '%%' en el string de formato del
    # logger, lo que producía un error de formato en tiempo de ejecución.
    # Se reemplaza con f-string para garantizar el formato correcto.
    logger.info(
        "Robustez: P(factible)=%.1f%% IC[%.1f%%, %.1f%%] dur_media=%.1fh p95=%.1fh",
        p_hat * 100, ci_lower * 100, ci_upper * 100, mean_dur, p95_dur,
    )

    return result


# ---------------------------------------------------------------------------
# Análisis de sensibilidad: P(factible) vs cv
# ---------------------------------------------------------------------------

def sensitivity_cv(
    problem,
    selected_ids: Sequence[str],
    cv_values: Optional[List[float]] = None,
    n_simulations: int = 3000,
    seed: int = 42,
) -> dict[float, float]:
    """
    Evalúa cómo varía P(factible) al cambiar el coeficiente de variación.
    Produce los datos para la tabla de sensibilidad del informe (Fase 4).

    Args:
        problem:       Instancia del problema.
        selected_ids:  Ruta a analizar (S*).
        cv_values:     Lista de valores de cv a probar.
                       Default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
        n_simulations: Simulaciones por valor de cv.
        seed:          Semilla para reproducibilidad.

    Returns:
        {cv: p_factible}
    """
    if cv_values is None:
        cv_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    results = {}
    for cv in cv_values:
        r = robustness_analysis(
            problem, selected_ids,
            n_simulations=n_simulations,
            cv=cv,
            seed=seed,
        )
        results[cv] = round(r.p_feasible, 4)
        logger.info("cv=%.2f -> P(factible)=%.1f%%", cv, r.p_feasible * 100)

    return results
