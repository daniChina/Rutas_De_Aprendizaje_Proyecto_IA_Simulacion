"""
problem.py
==========
Definición formal del problema de rutas de aprendizaje óptimas.

Modelado como un Grafo Dirigido Acíclico (DAG) G = (V, E) donde:
  - V : conjunto de cursos/recursos de aprendizaje.
  - E : aristas dirigidas que representan relaciones de prerrequisito.
        (v_i, v_j) ∈ E  →  v_i debe completarse antes que v_j.

El problema de optimización es:
    maximizar  Σ u(v) · x_v        para v ∈ S ⊆ V
    sujeto a   Σ d(v) · x_v ≤ T_max
               x_{v_j} ≤ x_{v_i}  para toda arista (v_i, v_j) ∈ E
               x_v ∈ {0, 1}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Nodo del grafo: un curso
# ---------------------------------------------------------------------------

@dataclass
class Course:
    """
    Representa un nodo del DAG: un curso o recurso de aprendizaje.

    Attributes:
        id:                 Identificador único (ej. "CS_302").
        titulo:             Nombre legible del curso.
        descripcion:        Texto en lenguaje natural para análisis semántico del LLM.
        duracion_horas:     Duración d(v) ∈ ℝ⁺ en horas.
        prerrequisitos:     Lista de IDs de cursos que deben completarse antes.
        utilidad_relativa:  Puntuación u(v) ∈ [1, 10] asignada por el LLM (None hasta Fase 2).
        justificacion:      Texto explicativo del LLM para el informe técnico.
    """
    id: str
    titulo: str
    descripcion: str
    duracion_horas: int
    prerrequisitos: List[str] = field(default_factory=list)

    # Campos poblados por el LLM en la Fase 2
    utilidad_relativa: Optional[int] = None
    justificacion: Optional[str] = None

    @property
    def duration(self) -> float:
        """Alias numérico de duracion_horas para compatibilidad con solvers."""
        return float(self.duracion_horas)

    @property
    def utility(self) -> float:
        """
        Devuelve la utilidad semántica u(v).
        Retorna 5.0 (neutral) si el LLM no ha evaluado el curso aún.
        """
        return float(self.utilidad_relativa) if self.utilidad_relativa is not None else 5.0

    def __repr__(self) -> str:
        u = f"u={self.utilidad_relativa}" if self.utilidad_relativa else "u=?"
        return f"Course({self.id!r}, d={self.duracion_horas}h, {u})"


# ---------------------------------------------------------------------------
# Problema: DAG + restricción de tiempo
# ---------------------------------------------------------------------------

@dataclass
class LearningPathProblem:
    """
    Encapsula el problema completo de selección de ruta de aprendizaje.

    Attributes:
        courses:  Lista ordenada de cursos (nodos del DAG).
        t_max:    Restricción dura de tiempo total en horas (T_max).
        instance_id: Identificador de la instancia de prueba (ej. "instancia_A").
    """
    courses: List[Course]
    t_max: float
    instance_id: str = "default"

    # Índice interno id → Course, construido post-init
    _index: Dict[str, Course] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {c.id: c for c in self.courses}

    # ------------------------------------------------------------------
    # Acceso y consulta
    # ------------------------------------------------------------------

    def get_course(self, course_id: str) -> Optional[Course]:
        """Devuelve el curso por ID o None si no existe en la instancia."""
        return self._index.get(course_id)

    @property
    def course_ids(self) -> List[str]:
        return [c.id for c in self.courses]

    @property
    def total_duration(self) -> float:
        """Duración acumulada de todos los cursos del problema (cota superior)."""
        return sum(c.duration for c in self.courses)

    # ------------------------------------------------------------------
    # Validación del DAG
    # ------------------------------------------------------------------

    def validate_dag(self) -> Tuple[bool, List[str]]:
        """
        Verifica que las dependencias forman un DAG válido (sin ciclos).
        Usa el algoritmo de Kahn (ordenamiento topológico por in-degree).

        Returns:
            (True, []) si el grafo es un DAG válido.
            (False, [lista de nodos en ciclo]) si se detecta algún ciclo.
        """
        in_degree: Dict[str, int] = {c.id: 0 for c in self.courses}
        adjacency: Dict[str, List[str]] = {c.id: [] for c in self.courses}

        # Solo considerar prerrequisitos que existan dentro de esta instancia
        for course in self.courses:
            for prereq_id in course.prerrequisitos:
                if prereq_id in self._index:
                    adjacency[prereq_id].append(course.id)
                    in_degree[course.id] += 1

        # Cola: nodos sin prerrequisitos dentro de la instancia
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        visited = 0

        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited == len(self.courses):
            return True, []

        # Nodos que quedaron con in_degree > 0 pertenecen a un ciclo
        cyclic_nodes = [cid for cid, deg in in_degree.items() if deg > 0]
        return False, cyclic_nodes

    def topological_order(self) -> List[str]:
        """
        Devuelve los IDs de los cursos en orden topológico válido.
        Útil para que el solver recorra el DAG de raíces a hojas.

        Raises:
            ValueError: Si el grafo contiene ciclos.
        """
        in_degree: Dict[str, int] = {c.id: 0 for c in self.courses}
        adjacency: Dict[str, List[str]] = {c.id: [] for c in self.courses}

        for course in self.courses:
            for prereq_id in course.prerrequisitos:
                if prereq_id in self._index:
                    adjacency[prereq_id].append(course.id)
                    in_degree[course.id] += 1

        queue = sorted([cid for cid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in sorted(adjacency[node]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.courses):
            raise ValueError(
                "El grafo contiene ciclos. No se puede calcular el orden topológico."
            )
        return order

    # ------------------------------------------------------------------
    # Validación de una solución propuesta
    # ------------------------------------------------------------------

    def is_valid_selection(self, selected_ids: Sequence[str]) -> bool:
        """
        Verifica que un subconjunto de cursos es una solución factible:
          1. Todos los IDs existen en la instancia.
          2. La suma de duraciones no supera T_max.
          3. La clausura de prerrequisitos se satisface (si v_j ∈ S → v_i ∈ S).

        Args:
            selected_ids: Secuencia de IDs de cursos seleccionados.

        Returns:
            True si la selección es válida, False en caso contrario.
        """
        selected_set: Set[str] = set(selected_ids)

        # Verificar existencia
        for cid in selected_set:
            if cid not in self._index:
                return False

        # Verificar restricción de tiempo
        total = sum(self._index[cid].duration for cid in selected_set)
        if total > self.t_max:
            return False

        # Verificar clausura de prerrequisitos
        for cid in selected_set:
            course = self._index[cid]
            for prereq_id in course.prerrequisitos:
                if prereq_id in self._index and prereq_id not in selected_set:
                    return False

        return True

    def objective_value(self, selected_ids: Sequence[str]) -> float:
        """
        Calcula el valor de la función objetivo para una selección dada:
            f(S) = Σ u(v) para v ∈ S

        Args:
            selected_ids: Secuencia de IDs de cursos seleccionados.

        Returns:
            Suma de utilidades de los cursos seleccionados.
        """
        return sum(self._index[cid].utility for cid in selected_ids if cid in self._index)

    def selection_duration(self, selected_ids: Sequence[str]) -> float:
        """Duración total de una selección de cursos."""
        return sum(self._index[cid].duration for cid in selected_ids if cid in self._index)

    def selected_courses(self, selected_ids: Sequence[str]) -> List[Course]:
        """Devuelve los objetos Course correspondientes a los IDs seleccionados."""
        return [self._index[cid] for cid in selected_ids if cid in self._index]

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Resumen legible de la instancia del problema."""
        is_dag, cycles = self.validate_dag()
        dag_status = "✓ DAG válido" if is_dag else f"✗ Ciclos en: {cycles}"
        edges = sum(
            1 for c in self.courses
            for p in c.prerrequisitos if p in self._index
        )
        all_evaluated = all(c.utilidad_relativa is not None for c in self.courses)

        return (
            f"Instancia : {self.instance_id}\n"
            f"Cursos    : {len(self.courses)} nodos, {edges} aristas\n"
            f"T_max     : {self.t_max:.0f} h\n"
            f"Duración  : {self.total_duration:.0f} h total en catálogo\n"
            f"DAG       : {dag_status}\n"
            f"LLM eval  : {'✓ Completa' if all_evaluated else '⏳ Pendiente (Fase 2)'}"
        )
