"""
mc_sampler.py  (src/solver/mc_sampler.py)
==========================================
Solver heurístico basado en simulación Monte Carlo para el problema de
selección de ruta de aprendizaje óptima con restricciones de precedencia en DAG.

Motivación:
    El problema es NP-hard (knapsack con precedencias en DAG). El DP exacto
    de baseline.py es correcto para instancias pequeñas (A, B), pero su
    complejidad exponencial lo hace prohibitivo en la Instancia C (35 nodos).
    Este módulo lo reemplaza con muestreo iterativo guiado por las utilidades
    del LLM: cursos con mayor u(v) tienen mayor probabilidad de ser seleccionados
    en cada paso, concentrando la búsqueda en regiones prometedoras.

Algoritmo: Monte Carlo Importance-Weighted Path Sampling
    Para cada iteración i = 1..N:
      1. S = ∅, budget = T_max, disponibles = nodos raíz del DAG.
      2. Mientras haya disponibles con d(v) <= budget:
         a. Calcular pesos softmax: w(v) = exp(u(v) / temperature).
         b. Muestrear v con probabilidad proporcional a w(v).
         c. Agregar v a S, descontar d(v) del budget.
         d. Actualizar disponibles: nodos cuyos prerrequisitos están todos en S.
      3. Si f(S) = Σu(v) > mejor hasta ahora, guardar S como S_best.
    Retornar S_best.

Complejidad: O(N · |V|²) — práctico para 35 nodos con N = 1000-5000.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado del solver
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """
    Resultado de una ejecución del solver Monte Carlo.

    Attributes:
        selected_ids:      IDs de cursos en la mejor ruta encontrada,
                           en orden topológico.
        objective_value:   Utilidad total Σu(v).
        total_duration:    Duración total Σd(v) en horas.
        n_iterations:      Iteraciones ejecutadas.
        n_feasible:        Iteraciones que produjeron solución no vacía.
        convergence_iter:  Iteración en que se encontró la mejor solución.
        utilidad_por_hora: Eficiencia de la ruta.
    """
    selected_ids: List[str]
    objective_value: float
    total_duration: float
    n_iterations: int
    n_feasible: int
    convergence_iter: int
    utilidad_por_hora: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.utilidad_por_hora = (
            self.objective_value / self.total_duration
            if self.total_duration > 0 else 0.0
        )

    def summary(self) -> str:
        return (
            f"MC Solver — {len(self.selected_ids)} cursos seleccionados\n"
            f"  Utilidad total  : {self.objective_value:.1f}\n"
            f"  Duracion total  : {self.total_duration:.0f} h\n"
            f"  Eficiencia      : {self.utilidad_por_hora:.3f} u/h\n"
            f"  Iteraciones     : {self.n_iterations} ({self.n_feasible} con solucion)\n"
            f"  Convergencia    : iteracion {self.convergence_iter}"
        )


# ---------------------------------------------------------------------------
# Tipos auxiliares (sin importar problem.py para poder testear en aislamiento)
# ---------------------------------------------------------------------------

class _CourseProxy:
    """Proxy mínimo de Course para el sampler (evita dependencia circular en tests)."""
    __slots__ = ("id", "duration", "utility", "prerrequisitos")

    def __init__(self, id: str, duration: float, utility: float, prereqs: List[str]) -> None:
        self.id = id
        self.duration = duration
        self.utility = utility
        self.prerrequisitos = prereqs


def _build_prereq_map(courses: list, index: Dict[str, "_CourseProxy"]) -> Dict[str, Set[str]]:
    """Solo incluye prerrequisitos que existan dentro de la instancia actual."""
    return {
        c.id: {p for p in c.prerrequisitos if p in index}
        for c in courses
    }


def _get_available(
    index: Dict[str, "_CourseProxy"],
    prereqs: Dict[str, Set[str]],
    selected: Set[str],
    budget: float,
) -> List[str]:
    return [
        cid for cid, course in index.items()
        if cid not in selected
        and prereqs[cid].issubset(selected)
        and course.duration <= budget
    ]


def _topo_sort(index: Dict[str, "_CourseProxy"], prereqs: Dict[str, Set[str]]) -> List[str]:
    """
    Orden topológico (Kahn) sobre el subgrafo definido por index y prereqs.
    Usado para devolver selected_ids en orden topológico en lugar de
    orden lexicográfico, que puede romper la semántica de precedencias.
    """
    in_degree: Dict[str, int] = {cid: 0 for cid in index}
    for cid, pset in prereqs.items():
        for p in pset:
            if p in index:
                in_degree[cid] += 1

    queue = [cid for cid, deg in in_degree.items() if deg == 0]
    result: List[str] = []

    while queue:
        # Orden determinista dentro del mismo nivel de precedencia
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        for cid in index:
            if node in prereqs[cid]:
                in_degree[cid] -= 1
                if in_degree[cid] == 0:
                    queue.append(cid)

    return result


# ---------------------------------------------------------------------------
# Solver principal
# ---------------------------------------------------------------------------

def mc_path_sampler(
    problem,                    # LearningPathProblem (evitar import circular)
    n_iterations: int = 1000,
    seed: Optional[int] = None,
    temperature: float = 1.0,
) -> MCResult:
    """
    Solver heurístico Monte Carlo para la selección óptima de rutas en el DAG.

    El parámetro `temperature` controla el balance exploración/explotación:
      - temperature → 0 : siempre elige el curso con mayor u(v) (greedy puro).
      - temperature = 1 : muestreo proporcional a exp(u(v))       (recomendado).
      - temperature > 1 : distribución más uniforme, más exploración.

    Distribución de muestreo (softmax ponderada):
        p(v) = exp(u(v) / T) / Σ exp(u(w) / T)   para w en disponibles

    Esto integra la señal semántica del LLM directamente en la distribución
    de muestreo: cursos evaluados con mayor utilidad son explorados primero.

    Args:
        problem:      Instancia LearningPathProblem con u(v) asignadas (Fase 2).
        n_iterations: Número de trayectorias Monte Carlo a simular.
        seed:         Semilla aleatoria para reproducibilidad.
        temperature:  Parámetro de exploración (default = 1.0).

    Returns:
        MCResult con la mejor ruta S* (en orden topológico) y métricas
        del proceso de búsqueda.

    Raises:
        ValueError: Si ningún curso tiene utilidad asignada (LLM no ejecutado).
    """
    if all(c.utilidad_relativa is None for c in problem.courses):
        raise ValueError(
            "Ningún curso tiene utilidad asignada. "
            "Ejecuta primero la evaluación LLM (Fase 2) antes del solver."
        )

    if seed is not None:
        random.seed(seed)

    # Construir índice id → proxy
    index: Dict[str, _CourseProxy] = {
        c.id: _CourseProxy(c.id, c.duration, c.utility, c.prerrequisitos)
        for c in problem.courses
    }
    prereqs = _build_prereq_map(list(index.values()), index)

    # Orden topológico completo del grafo — usado para ordenar la selección final
    topo_order = _topo_sort(index, prereqs)

    best_set: Set[str] = set()
    best_value: float = -1.0
    n_feasible: int = 0
    convergence_iter: int = 0

    logger.info(
        "MC Sampler: instancia=%s | N=%d | T_max=%.0f h | temperature=%.2f",
        problem.instance_id, n_iterations, problem.t_max, temperature,
    )

    for iteration in range(1, n_iterations + 1):
        selected: Set[str] = set()
        budget: float = problem.t_max
        available = _get_available(index, prereqs, selected, budget)

        # Construcción iterativa greedy-estocástica de la ruta
        while available:
            utilities = [index[cid].utility for cid in available]
            weights = [math.exp(u / temperature) for u in utilities]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]

            chosen = random.choices(available, weights=probs, k=1)[0]
            selected.add(chosen)
            budget -= index[chosen].duration
            available = _get_available(index, prereqs, selected, budget)

        if selected:
            n_feasible += 1

        current_value = sum(index[cid].utility for cid in selected)

        if current_value > best_value:
            best_value = current_value
            best_set = selected.copy()
            convergence_iter = iteration
            logger.debug(
                "Mejor en iter %d: %d cursos u=%.1f", iteration, len(best_set), best_value
            )

    # CORRECCIÓN: devolver IDs en orden topológico, no lexicográfico.
    # sorted(selected) producía un orden arbitrario que podía romper la
    # semántica de precedencias esperada por los callers.
    best_ids = [cid for cid in topo_order if cid in best_set]
    best_duration = sum(index[cid].duration for cid in best_ids)

    logger.info(
        "MC completado: u=%.1f | %.0f/%.0fh | conv.iter=%d",
        best_value, best_duration, problem.t_max, convergence_iter,
    )

    return MCResult(
        selected_ids=best_ids,
        objective_value=best_value,
        total_duration=best_duration,
        n_iterations=n_iterations,
        n_feasible=n_feasible,
        convergence_iter=convergence_iter,
    )


# ---------------------------------------------------------------------------
# Análisis de convergencia (para gráfico del informe)
# ---------------------------------------------------------------------------

def convergence_analysis(
    problem,
    n_iterations: int = 2000,
    checkpoints: Optional[List[int]] = None,
    seed: int = 42,
    temperature: float = 1.0,
) -> Dict[int, float]:
    """
    Ejecuta el sampler registrando la mejor utilidad en cada checkpoint.
    Produce los datos para el gráfico de convergencia del informe (Fase 4).

    Args:
        problem:      Instancia con utilidades asignadas.
        n_iterations: Total de iteraciones.
        checkpoints:  Iteraciones en que se registra el mejor valor encontrado.
        seed:         Semilla para reproducibilidad.
        temperature:  Temperatura del muestreo softmax (default 1.0).
                      CORRECCIÓN: antes se ignoraba este parámetro y siempre
                      se usaba temperature=1.0 implícitamente (math.exp(u)).
                      Ahora es consistente con mc_path_sampler.

    Returns:
        {iteracion: mejor_utilidad_acumulada_hasta_ese_punto}
    """
    if checkpoints is None:
        defaults = [50, 100, 200, 500, 1000, 2000]
        checkpoints = [c for c in defaults if c <= n_iterations]
        if n_iterations not in checkpoints:
            checkpoints.append(n_iterations)

    random.seed(seed)
    index: Dict[str, _CourseProxy] = {
        c.id: _CourseProxy(c.id, c.duration, c.utility, c.prerrequisitos)
        for c in problem.courses
    }
    prereqs = _build_prereq_map(list(index.values()), index)

    best_value: float = 0.0
    results: Dict[int, float] = {}
    checkpoints_set = set(checkpoints)

    for iteration in range(1, n_iterations + 1):
        selected: Set[str] = set()
        budget: float = problem.t_max
        available = _get_available(index, prereqs, selected, budget)

        while available:
            utilities = [index[cid].utility for cid in available]
            # CORRECCIÓN: usar el parámetro temperature en lugar de exp(u) fijo.
            weights = [math.exp(u / temperature) for u in utilities]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen = random.choices(available, weights=probs, k=1)[0]
            selected.add(chosen)
            budget -= index[chosen].duration
            available = _get_available(index, prereqs, selected, budget)

        current = sum(index[cid].utility for cid in selected)
        best_value = max(best_value, current)

        if iteration in checkpoints_set:
            results[iteration] = round(best_value, 2)

    return results
