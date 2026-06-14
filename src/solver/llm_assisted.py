"""
llm_assisted.py — Fase 3
========================
Solver híbrido: usa puntuaciones de utilidad del LLM como función objetivo
del algoritmo clásico de optimización.

Flujo completo del sistema híbrido:
    cursos.json
        ↓  (Fase 1) load_instance()
    LearningPathProblem
        ↓  (Fase 2) evaluar_problema() → u(v) ∈ [1, 10]
    LearningPathProblem con utilidades
        ↓  (Fase 3) llm_assisted_solver()
    Ruta óptima S* ⊆ V  +  métricas comparativas

El solver híbrido orquesta tres estrategias y devuelve la mejor:
    1. DP exacto         — óptimo garantizado para instancias A y B.
    2. Greedy densidad   — heurística rápida, buen punto de partida.
    3. Monte Carlo       — muestreo estocástico, robusto para instancia C.

El resultado final siempre incluye el DP como referencia exacta cuando
es tratable (n × W ≤ MAX_DP_STATES), y Monte Carlo como respaldo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..problem import LearningPathProblem
from ..llm.client import LLMClient
from ..llm.evaluator import evaluar_problema
from .baseline import DPResult, dp_knapsack_dag, greedy_by_utility_density
from .mc_sampler import MCResult, mc_path_sampler

logger = logging.getLogger(__name__)

# Umbral de estados DP por encima del cual se omite el DP exacto en favor de MC.
# 35 nodos × 300 h = 10 500, muy por debajo del límite de 500 000.
MAX_DP_STATES = 500_000


# ---------------------------------------------------------------------------
# Resultado del solver híbrido
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """
    Resultado de una ejecución del solver híbrido LLM + DP/MC.

    Attributes:
        selected_ids:       IDs de la mejor ruta encontrada.
        objective_value:    Utilidad total Σu(v) de la mejor ruta.
        total_duration:     Duración total Σd(v) en horas.
        solver_used:        Nombre del solver que produjo la mejor ruta.
        dp_result:          Resultado del DP exacto (None si se omitió).
        greedy_result:      Resultado del greedy (siempre disponible).
        mc_result:          Resultado del Monte Carlo (siempre disponible).
        utilidad_por_hora:  Eficiencia de la mejor ruta.
    """
    selected_ids: List[str]
    objective_value: float
    total_duration: float
    solver_used: str
    dp_result: Optional[DPResult]
    greedy_result: DPResult
    mc_result: MCResult
    utilidad_por_hora: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.utilidad_por_hora = (
            self.objective_value / self.total_duration
            if self.total_duration > 0 else 0.0
        )

    def summary(self) -> str:
        lines = [
            f"Hybrid Solver — mejor solución por: {self.solver_used}",
            f"  Cursos seleccionados : {len(self.selected_ids)}",
            f"  Utilidad total       : {self.objective_value:.1f}",
            f"  Duración total       : {self.total_duration:.0f} h",
            f"  Eficiencia           : {self.utilidad_por_hora:.3f} u/h",
            "",
            "  ── Comparativa de solvers ──",
        ]
        if self.dp_result:
            lines.append(
                f"  DP exacto   : u={self.dp_result.objective_value:.1f}"
                f"  ({len(self.dp_result.selected_ids)} cursos,"
                f" {self.dp_result.total_duration:.0f} h)"
            )
        else:
            lines.append("  DP exacto   : omitido (problema demasiado grande)")
        lines.append(
            f"  Greedy      : u={self.greedy_result.objective_value:.1f}"
            f"  ({len(self.greedy_result.selected_ids)} cursos,"
            f" {self.greedy_result.total_duration:.0f} h)"
        )
        lines.append(
            f"  Monte Carlo : u={self.mc_result.objective_value:.1f}"
            f"  ({len(self.mc_result.selected_ids)} cursos,"
            f" {self.mc_result.total_duration:.0f} h,"
            f" {self.mc_result.n_iterations} iter.)"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solver híbrido principal
# ---------------------------------------------------------------------------

def llm_assisted_solver(
    problem: LearningPathProblem,
    objetivo_usuario: str,
    *,
    llm_client: Optional[LLMClient] = None,
    mc_iterations: int = 2000,
    mc_temperature: float = 1.0,
    mc_seed: Optional[int] = 42,
    granularidad_dp: int = 1,
    force_mc_only: bool = False,
) -> HybridResult:
    """
    Solver híbrido LLM + DP/MC:

      1. Evaluación semántica (Fase 2):
         Si algún curso carece de utilidad, llama a evaluar_problema() con
         el LLM para asignar u(v) ∈ [1, 10] a cada curso.

      2. Solver DP exacto (Fase 3a):
         Si el espacio de estados es tratable (n × W ≤ MAX_DP_STATES) y no
         se fuerza MC, ejecuta dp_knapsack_dag() para la solución óptima.

      3. Solver greedy:
         Siempre ejecuta greedy_by_utility_density() como línea base rápida.

      4. Solver Monte Carlo (Fase 3):
         Siempre ejecuta mc_path_sampler() con los parámetros indicados.

      5. Selección de la mejor ruta:
         Devuelve la ruta con mayor utilidad entre DP, greedy y MC.

    Args:
        problem:           Instancia del problema (con o sin utilidades).
        objetivo_usuario:  Texto libre que describe el objetivo de aprendizaje.
                           Usado por el LLM para asignar utilidades semánticas.
        llm_client:        Cliente LLM preconfigurado. Si es None y hay cursos
                           sin evaluar, se construye uno por defecto.
        mc_iterations:     Número de iteraciones Monte Carlo (default 2 000).
        mc_temperature:    Temperatura del muestreo softmax (default 1.0).
        mc_seed:           Semilla aleatoria para reproducibilidad (default 42).
        granularidad_dp:   Resolución temporal del DP en horas (default 1 h).
        force_mc_only:     Si True, omite el DP exacto incluso si es tratable.

    Returns:
        HybridResult con la mejor ruta, métricas comparativas de los tres
        solvers y el nombre del solver que produjo el resultado.

    Raises:
        ValueError: Si el DAG contiene ciclos.
    """
    # ---------------------------------------------------------------
    # 1. Evaluación semántica — garantiza u(v) para todos los cursos
    # ---------------------------------------------------------------
    needs_eval = any(c.utilidad_relativa is None for c in problem.courses)

    if needs_eval:
        logger.info(
            "Instancia '%s': %d cursos sin utilidad. Iniciando evaluación LLM...",
            problem.instance_id,
            sum(1 for c in problem.courses if c.utilidad_relativa is None),
        )
        problem = evaluar_problema(
            problem,
            objetivo_usuario,
            llm_client=llm_client,
        )
        logger.info("Evaluación LLM completada para '%s'.", problem.instance_id)
    else:
        logger.info(
            "Instancia '%s': todos los cursos ya tienen utilidad asignada. "
            "Saltando evaluación LLM.",
            problem.instance_id,
        )

    # ---------------------------------------------------------------
    # 2. Decidir si DP exacto es tratable
    # ---------------------------------------------------------------
    n = len(problem.courses)
    W = int(problem.t_max // granularidad_dp)
    dp_tractable = (not force_mc_only) and (n * W <= MAX_DP_STATES)

    dp_result: Optional[DPResult] = None

    if dp_tractable:
        logger.info(
            "Ejecutando DP exacto (n=%d, W=%d, estados=%d)...", n, W, n * W
        )
        try:
            dp_result = dp_knapsack_dag(problem, granularidad=granularidad_dp)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("DP exacto falló (%s). Continuando sin él.", exc)
    else:
        reason = "force_mc_only=True" if force_mc_only else f"n×W={n*W} > {MAX_DP_STATES}"
        logger.info("DP exacto omitido (%s). Usando MC como solver principal.", reason)

    # ---------------------------------------------------------------
    # 3. Greedy
    # ---------------------------------------------------------------
    logger.info("Ejecutando greedy por densidad de utilidad...")
    greedy_result = greedy_by_utility_density(problem)

    # ---------------------------------------------------------------
    # 4. Monte Carlo
    # ---------------------------------------------------------------
    logger.info(
        "Ejecutando Monte Carlo (%d iter., temperature=%.2f, seed=%s)...",
        mc_iterations, mc_temperature, mc_seed,
    )
    mc_result = mc_path_sampler(
        problem,
        n_iter=mc_iterations,
        temperature=mc_temperature,
        seed=mc_seed,
    )

    # ---------------------------------------------------------------
    # 5. Elegir la mejor solución entre los tres solvers
    # ---------------------------------------------------------------
    candidates: List[Tuple[float, str, List[str]]] = [
        (greedy_result.objective_value, "greedy", greedy_result.selected_ids),
        (mc_result.objective_value, "monte_carlo", mc_result.selected_ids),
    ]
    if dp_result is not None:
        candidates.append(
            (dp_result.objective_value, "dp_exacto", dp_result.selected_ids)
        )

    best_value, best_solver, best_ids = max(candidates, key=lambda t: t[0])

    logger.info(
        "Mejor solver: %s  (utilidad=%.1f, cursos=%d)",
        best_solver, best_value, len(best_ids),
    )

    return HybridResult(
        selected_ids=best_ids,
        objective_value=best_value,
        total_duration=problem.selection_duration(best_ids),
        solver_used=best_solver,
        dp_result=dp_result,
        greedy_result=greedy_result,
        mc_result=mc_result,
    )
