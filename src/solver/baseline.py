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
- Para instancias pequeñas (n ≤ 20): DP exacta por máscara de bits (2^n estados).
- Para instancias grandes (n > 20): DP heurística con estado (i, w) y clausura
  de prerrequisitos post‑backtracking (no garantiza optimalidad pero es rápida).

Complejidad exacta: O(2^n) para n ≤ 20, aceptable para tests y casos pequeños.
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
# DP exacta por máscara de bits (para n ≤ 20)
# ---------------------------------------------------------------------------

def _dp_bitmask(
    problem: LearningPathProblem,
    granularidad: int = 1,
) -> DPResult:
    """
    Solver DP exacto usando máscara de bits (2^n estados).
    Solo factible para n ≤ 20 (unos ~1M estados como máximo).
    """
    n = len(problem.courses)
    cursos = list(problem.courses)
    # Mapeo id -> índice
    id_to_idx = {c.id: i for i, c in enumerate(cursos)}
    # Máscara de prerrequisitos para cada curso
    prereq_mask = [0] * n
    for i, c in enumerate(cursos):
        mask = 0
        for p in c.prerrequisitos:
            if p in id_to_idx:
                mask |= (1 << id_to_idx[p])
        prereq_mask[i] = mask
    # Duración y utilidad discretizadas
    duraciones = [int(c.duration // granularidad) for c in cursos]
    utilidades = [c.utility for c in cursos]
    W = int(problem.t_max // granularidad)
    if W <= 0:
        return DPResult([], 0.0, 0.0, n, 0)

    # dp[máscara] = (utilidad, duración)
    dp = {0: (0.0, 0)}
    best_mask = 0
    best_util = -1.0

    # Itera sobre todas las máscaras posibles (2^n)
    for mask in range(1 << n):
        if mask not in dp:
            continue
        u_act, d_act = dp[mask]
        # Intenta agregar cada curso no seleccionado
        for i in range(n):
            if mask & (1 << i):
                continue
            # Verifica prerrequisitos
            if (prereq_mask[i] & mask) != prereq_mask[i]:
                continue
            new_d = d_act + duraciones[i]
            if new_d <= W:
                new_mask = mask | (1 << i)
                new_u = u_act + utilidades[i]
                # Guarda si mejora
                if new_mask not in dp or new_u > dp[new_mask][0]:
                    dp[new_mask] = (new_u, new_d)
                    if new_u > best_util:
                        best_util = new_u
                        best_mask = new_mask

    # Reconstruir IDs en orden topológico
    selected_ids = []
    for i in range(n):
        if best_mask & (1 << i):
            selected_ids.append(cursos[i].id)
    topo = problem.topological_order()
    selected_ids = [cid for cid in topo if cid in selected_ids]

    obj_value = problem.objective_value(selected_ids)
    total_dur = problem.selection_duration(selected_ids)

    logger.info(
        "DP exacto (bitmask) completado: %d cursos, utilidad=%.1f, duración=%.0f h / %.0f h",
        len(selected_ids), obj_value, total_dur, problem.t_max,
    )
    return DPResult(
        selected_ids=selected_ids,
        objective_value=obj_value,
        total_duration=total_dur,
        n_nodes=n,
        n_states=len(dp),
    )


# ---------------------------------------------------------------------------
# DP heurística para instancias grandes (n > 20)
# ---------------------------------------------------------------------------

def _dp_heuristic(
    problem: LearningPathProblem,
    granularidad: int = 1,
) -> DPResult:
    """
    Versión heurística (no óptima) para problemas con n > 20.
    Usa estado (i, w) y corrige prerrequisitos después del backtracking.
    """
    is_dag, cycles = problem.validate_dag()
    if not is_dag:
        raise ValueError(f"El grafo contiene ciclos en los nodos: {cycles}")

    topo_order: List[str] = problem.topological_order()
    n = len(topo_order)
    W = int(problem.t_max // granularidad)
    if W <= 0:
        logger.warning(
            "T_max=%s con granularidad=%s produce W=0. Retornando selección vacía.",
            problem.t_max, granularidad,
        )
        return DPResult([], 0.0, 0.0, n, 0)

    courses_by_pos = [problem.get_course(cid) for cid in topo_order]
    pos_of = {cid: i for i, cid in enumerate(topo_order)}

    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    included = [[False] * (W + 1) for _ in range(n + 1)]
    n_states = 0

    for i, course in enumerate(courses_by_pos, start=1):
        d_i = int(course.duration // granularidad)
        u_i = course.utility
        for w in range(W + 1):
            n_states += 1
            best = dp[i - 1][w]
            take = False
            if d_i <= w:
                val_include = dp[i - 1][w - d_i] + u_i
                if val_include > best:
                    best = val_include
                    take = True
            dp[i][w] = best
            included[i][w] = take

    # Reconstrucción
    candidate_set = set()
    w_rem = W
    for i in range(n, 0, -1):
        course = courses_by_pos[i - 1]
        d_i = int(course.duration // granularidad)
        if included[i][w_rem] and w_rem >= d_i:
            candidate_set.add(course.id)
            w_rem -= d_i

    # Clausura de prerrequisitos
    selected_set = set()
    for cid in topo_order:
        if cid not in candidate_set:
            continue
        course = problem.get_course(cid)
        prereqs_ok = all(
            p not in pos_of or p in selected_set
            for p in course.prerrequisitos
        )
        if prereqs_ok:
            selected_set.add(cid)

    selected_ids = [cid for cid in topo_order if cid in selected_set]
    obj_value = problem.objective_value(selected_ids)
    total_dur = problem.selection_duration(selected_ids)

    logger.info(
        "DP heurístico completado: %d cursos, utilidad=%.1f, duración=%.0f h / %.0f h",
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
# Función principal dp_knapsack_dag (pública)
# ---------------------------------------------------------------------------

def dp_knapsack_dag(
    problem: LearningPathProblem,
    granularidad: int = 1,
) -> DPResult:
    """
    Solver DP para el problema de la mochila con precedencias (DAG).
    - Si el número de cursos ≤ 20, usa DP exacta por máscara de bits (óptima).
    - Si > 20, usa versión heurística rápida (no garantiza optimalidad).

    Args:
        problem:       Instancia del problema.
        granularidad:  Resolución temporal en horas (default 1).

    Returns:
        DPResult con la mejor selección encontrada.
    """
    n = len(problem.courses)
    if n <= 20:
        logger.info("Usando DP exacta por bitmask (n=%d <= 20)", n)
        return _dp_bitmask(problem, granularidad)
    else:
        logger.info("Usando DP heurística (n=%d > 20)", n)
        return _dp_heuristic(problem, granularidad)


# ---------------------------------------------------------------------------
# Heurística greedy por densidad de utilidad
# ---------------------------------------------------------------------------

def greedy_by_utility_density(
    problem: LearningPathProblem,
) -> DPResult:
    """
    Heurística greedy: ordena cursos por u(v)/d(v) descendente y los
    selecciona respetando la clausura de prerrequisitos y el presupuesto.
    """
    is_dag, cycles = problem.validate_dag()
    if not is_dag:
        raise ValueError(f"El grafo contiene ciclos en los nodos: {cycles}")

    all_course_ids = {c.id for c in problem.courses}
    candidates = sorted(
        problem.courses,
        key=lambda c: c.utility / c.duration if c.duration > 0 else 0.0,
        reverse=True,
    )
    selected = set()
    budget_remaining = problem.t_max
    improved = True
    while improved:
        improved = False
        for course in candidates:
            if course.id in selected:
                continue
            if course.duration > budget_remaining:
                continue
            prereqs_ok = all(
                p not in all_course_ids or p in selected
                for p in course.prerrequisitos
            )
            if prereqs_ok:
                selected.add(course.id)
                budget_remaining -= course.duration
                improved = True

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