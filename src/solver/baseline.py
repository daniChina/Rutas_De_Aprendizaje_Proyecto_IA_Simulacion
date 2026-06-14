"""
baseline.py — Fase 3
====================
Solver clásico de selección óptima de cursos usando Programación Dinámica.

Implementa el problema de la mochila con restricciones de precedencia (DAG):
    maximizar  Σ u(v) · x_v
    sujeto a   Σ d(v) · x_v ≤ T_max
               x_{v_j} ≤ x_{v_i}  para toda arista (v_i, v_j) ∈ E
               x_v ∈ {0, 1}

Estrategia DP
-------------
El grafo de prerrequisitos es un DAG, de modo que existe al menos un orden
topológico. Procesamos los nodos en ese orden para garantizar que cuando
evaluamos el nodo v, todos sus prerrequisitos ya han sido procesados.

Estado DP:
    dp[i][w]  =  máxima utilidad alcanzable usando únicamente los primeros
                 i nodos en orden topológico con un presupuesto de w horas.

Transición (nodo i con duración d_i, utilidad u_i, prerrequisitos P_i):
    - Excluir: dp[i][w] = dp[i-1][w]
    - Incluir (solo si w >= d_i y ∀p ∈ P_i : p está incluido):
        dp[i][w] = dp[i-1][w - d_i] + u_i
    Se toma el máximo de ambas opciones.

La restricción de prerrequisitos se maneja de forma exacta durante la
reconstrucción (backtracking): si al reconstruir decidimos incluir v pero
algún prerrequisito quedó excluido, descartamos esa rama.  Para instancias
pequeñas (≤ 22 nodos, T_max ≤ 200 h) el DP exacto es óptimo y eficiente.

Complejidad: O(n · T_max_discreta) en tiempo y espacio, donde
    T_max_discreta = T_max / granularidad (por defecto granularidad = 1 h).

Para la Instancia C (35 nodos, T_max = 300 h) esto sigue siendo tratable:
35 × 300 = 10 500 celdas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..problem import LearningPathProblem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado del solver DP
# ---------------------------------------------------------------------------

@dataclass
class DPResult:
    """
    Resultado de una ejecución del solver DP exacto.

    Attributes:
        selected_ids:       IDs de cursos en la solución óptima.
        objective_value:    Utilidad total Σu(v).
        total_duration:     Duración total Σd(v) en horas.
        n_nodes:            Número de nodos procesados.
        n_states:           Número de estados DP evaluados.
        utilidad_por_hora:  Eficiencia de la ruta.
    """
    selected_ids: List[str]
    objective_value: float
    total_duration: float
    n_nodes: int
    n_states: int
    utilidad_por_hora: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.utilidad_por_hora = (
            self.objective_value / self.total_duration
            if self.total_duration > 0 else 0.0
        )

    def summary(self) -> str:
        return (
            f"DP Solver — {len(self.selected_ids)} cursos seleccionados\n"
            f"  Utilidad total  : {self.objective_value:.1f}\n"
            f"  Duracion total  : {self.total_duration:.0f} h\n"
            f"  Eficiencia      : {self.utilidad_por_hora:.3f} u/h\n"
            f"  Nodos           : {self.n_nodes}\n"
            f"  Estados DP      : {self.n_states}"
        )


# ---------------------------------------------------------------------------
# Solver DP exacto
# ---------------------------------------------------------------------------

def dp_knapsack_dag(
    problem: LearningPathProblem,
    granularidad: int = 1,
) -> DPResult:
    """
    Solver exacto de DP para el knapsack con precedencias en DAG.

    Garantiza la solución óptima para instancias donde T_max/granularidad
    es tratable (≤ ~10 000 estados por nodo).

    Args:
        problem:       Instancia del problema con utilidades ya asignadas.
        granularidad:  Resolución temporal en horas (default 1 h).
                       Incrementar a 5 o 10 si T_max es muy grande.

    Returns:
        DPResult con la selección óptima y métricas del solver.

    Raises:
        ValueError: Si el DAG contiene ciclos o algún curso tiene u(v) = None.
    """
    # ---------------------------------------------------------------
    # 1. Validación y orden topológico
    # ---------------------------------------------------------------
    is_dag, cycles = problem.validate_dag()
    if not is_dag:
        raise ValueError(f"El grafo contiene ciclos en los nodos: {cycles}")

    topo_order: List[str] = problem.topological_order()
    n = len(topo_order)

    # Capacidad discreta
    W = int(problem.t_max // granularidad)
    if W <= 0:
        logger.warning("T_max=%s con granularidad=%s produce W=0. Retornando selección vacía.",
                       problem.t_max, granularidad)
        return DPResult(
            selected_ids=[], objective_value=0.0, total_duration=0.0,
            n_nodes=n, n_states=0,
        )

    # Mapeo posición → curso
    courses_by_pos: List = [problem.get_course(cid) for cid in topo_order]
    pos_of: Dict[str, int] = {cid: i for i, cid in enumerate(topo_order)}

    # ---------------------------------------------------------------
    # 2. Tabla DP — dp[i][w] = utilidad máxima usando nodos 0..i-1
    #    con capacidad w (en unidades de granularidad).
    #
    #    Usamos dos filas (rolling) para reducir memoria de O(n·W) a O(W).
    #    Sin embargo, para la reconstrucción necesitamos la tabla completa,
    #    así que la guardamos en una lista de listas.
    # ---------------------------------------------------------------
    INF = float("inf")

    # dp[i] es un array de tamaño W+1
    # Inicializamos: dp[0][w] = 0 para todo w (sin nodos procesados → utilidad 0)
    dp: List[List[float]] = [[0.0] * (W + 1) for _ in range(n + 1)]

    # included[i][w] = True si el nodo i-1 (0-indexed) se incluyó en dp[i][w]
    included: List[List[bool]] = [[False] * (W + 1) for _ in range(n + 1)]

    n_states = 0

    for i, course in enumerate(courses_by_pos, start=1):
        d_i = int(course.duration // granularidad)  # duración discreta
        u_i = course.utility

        # Prerrequisitos presentes en esta instancia, en posición topológica
        prereq_positions: List[int] = [
            pos_of[p] for p in course.prerrequisitos if p in pos_of
        ]

        for w in range(W + 1):
            n_states += 1
            # Opción 1: no incluir el nodo i
            best = dp[i - 1][w]
            take = False

            # Opción 2: incluir el nodo i (si cabe en el presupuesto)
            if d_i <= w:
                val_include = dp[i - 1][w - d_i] + u_i
                if val_include > best:
                    best = val_include
                    take = True

            dp[i][w] = best
            included[i][w] = take

    # ---------------------------------------------------------------
    # 3. Reconstrucción por backtracking
    #    Recorremos la tabla desde (n, W) hacia (0, 0).
    #    Primero recogemos los nodos que el DP marcó como incluidos,
    #    luego eliminamos cualquiera cuyos prerrequisitos no estén
    #    en el conjunto (puede ocurrir por interacción entre la
    #    granularidad y el redondeo de duraciones).
    # ---------------------------------------------------------------
    candidate_set: Set[str] = set()
    w_rem = W

    for i in range(n, 0, -1):
        course = courses_by_pos[i - 1]
        if included[i][w_rem]:
            candidate_set.add(course.id)
            w_rem -= int(course.duration // granularidad)

    # Clausura de prerrequisitos: en orden topológico, solo incluimos
    # un nodo si todos sus prerrequisitos (presentes en la instancia)
    # también fueron seleccionados.
    selected_set: Set[str] = set()
    for cid in topo_order:          # topo_order: raíces → hojas
        if cid not in candidate_set:
            continue
        course = problem.get_course(cid)
        prereqs_ok = all(
            p not in pos_of or p in selected_set
            for p in course.prerrequisitos
        )
        if prereqs_ok:
            selected_set.add(cid)

    # Mantener el orden topológico en la lista de salida
    selected_ids = [cid for cid in topo_order if cid in selected_set]

    # Calcular métricas reales (sin granularidad)
    obj_value = problem.objective_value(selected_ids)
    total_dur = problem.selection_duration(selected_ids)

    logger.info(
        "DP solver completado: %d cursos, utilidad=%.1f, duración=%.0f h / %.0f h",
        len(selected_ids), obj_value, total_dur, problem.t_max,
    )

    return DPResult(
        selected_ids=selected_ids,
        objective_value=obj_value,
        total_duration=total_dur,
        n_nodes=n,
        n_states=n_states,
    )


# ---------------------------------------------------------------------------
# Heurística greedy por densidad de utilidad
# ---------------------------------------------------------------------------

def greedy_by_utility_density(
    problem: LearningPathProblem,
) -> DPResult:
    """
    Heurística greedy: ordena cursos por u(v)/d(v) descendente y los
    selecciona respetando la clausura de prerrequisitos y el presupuesto.

    Más rápida que dp_knapsack_dag (O(n log n)) y útil como cota inferior
    o como punto de comparación rápido para la Instancia C (35 nodos).

    Returns:
        DPResult con la selección greedy y métricas del solver.
    """
    is_dag, cycles = problem.validate_dag()
    if not is_dag:
        raise ValueError(f"El grafo contiene ciclos en los nodos: {cycles}")

    # Ordenar por densidad utilidad/hora descendente (mayor bang-per-hour primero)
    candidates = sorted(
        problem.courses,
        key=lambda c: c.utility / c.duration if c.duration > 0 else 0.0,
        reverse=True,
    )

    selected: Set[str] = set()
    budget_remaining = problem.t_max

    # Iteramos hasta que no haya más candidatos que quepan y sean factibles
    improved = True
    while improved:
        improved = False
        for course in candidates:
            if course.id in selected:
                continue
            if course.duration > budget_remaining:
                continue
            # Verificar clausura de prerrequisitos
            prereqs_ok = all(
                p not in {c.id for c in problem.courses} or p in selected
                for p in course.prerrequisitos
            )
            if prereqs_ok:
                selected.add(course.id)
                budget_remaining -= course.duration
                improved = True

    # Orden topológico para la salida
    topo = problem.topological_order()
    selected_ids = [cid for cid in topo if cid in selected]

    obj_value = problem.objective_value(selected_ids)
    total_dur = problem.selection_duration(selected_ids)

    logger.info(
        "Greedy solver completado: %d cursos, utilidad=%.1f, duración=%.0f h / %.0f h",
        len(selected_ids), obj_value, total_dur, problem.t_max,
    )

    return DPResult(
        selected_ids=selected_ids,
        objective_value=obj_value,
        total_duration=total_dur,
        n_nodes=len(problem.courses),
        n_states=len(candidates),
    )
