"""
baseline.py — Fase 3
====================
Solver clásico de selección óptima de cursos usando Programación Dinámica.

Implementa el problema de la mochila con restricciones de precedencia (DAG):
    maximizar  Σ u(v) · x_v
    sujeto a   Σ d(v) · x_v ≤ T_max
               x_{v_j} ≤ x_{v_i}  para toda arista (v_i, v_j) ∈ E
               x_v ∈ {0, 1}

Estado: STUB — implementación completa en la Fase 3.
"""

from __future__ import annotations

from typing import List, Tuple

# TODO (Fase 3): importar LearningPathProblem
# from ..problem import LearningPathProblem


def dp_knapsack_dag(
    # problem: LearningPathProblem,
) -> Tuple[List[str], float]:
    """
    Solver exacto de DP para el knapsack con precedencias en DAG.

    Estrategia:
      1. Obtener orden topológico del DAG.
      2. Usar DP sobre los nodos en ese orden, respetando la clausura de prerrequisitos.
      3. Devolver la selección de IDs óptima y su valor objetivo.

    TODO (Fase 3): implementar con:
      - Estado DP: (índice_topológico, capacidad_restante)
      - Transición: incluir/excluir nodo si sus prerrequisitos ya están en S
      - Reconstrucción de la solución por backtracking

    Returns:
        (lista_de_ids_seleccionados, valor_objetivo_total)
    """
    # PLACEHOLDER
    raise NotImplementedError("Solver DP — implementar en Fase 3.")


def greedy_by_utility_density(
    # problem: LearningPathProblem,
) -> Tuple[List[str], float]:
    """
    Heurística greedy: ordena cursos por u(v)/d(v) y selecciona con clausura de prerrequisitos.
    Más rápida que DP exacto, útil para la Instancia C (35 nodos).

    TODO (Fase 3): implementar.

    Returns:
        (lista_de_ids_seleccionados, valor_objetivo_total)
    """
    # PLACEHOLDER
    raise NotImplementedError("Solver greedy — implementar en Fase 3.")
