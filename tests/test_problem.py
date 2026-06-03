"""
test_problem.py
===============
Pruebas unitarias para src/problem.py e src/instance.py.

Ejecutar desde la raíz del proyecto:
    python -m pytest tests/ -v
    # o sin pytest:
    python tests/test_problem.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.problem import Course, LearningPathProblem


# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

def make_linear_problem() -> LearningPathProblem:
    """Grafo lineal: A → B → C (sin ramificaciones)."""
    return LearningPathProblem(
        courses=[
            Course(id="A", titulo="Curso A", descripcion="Base", duracion_horas=10, prerrequisitos=[]),
            Course(id="B", titulo="Curso B", descripcion="Intermedio", duracion_horas=20, prerrequisitos=["A"]),
            Course(id="C", titulo="Curso C", descripcion="Avanzado", duracion_horas=30, prerrequisitos=["B"]),
        ],
        t_max=50,
        instance_id="test_lineal",
    )


def make_branched_problem() -> LearningPathProblem:
    """
    Grafo con ramificación:
        A ──► C
        B ──► C
        C ──► D
    """
    return LearningPathProblem(
        courses=[
            Course(id="A", titulo="A", descripcion="x", duracion_horas=10, prerrequisitos=[]),
            Course(id="B", titulo="B", descripcion="x", duracion_horas=15, prerrequisitos=[]),
            Course(id="C", titulo="C", descripcion="x", duracion_horas=20, prerrequisitos=["A", "B"]),
            Course(id="D", titulo="D", descripcion="x", duracion_horas=10, prerrequisitos=["C"]),
        ],
        t_max=60,
        instance_id="test_ramificado",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dag_valido_lineal():
    problem = make_linear_problem()
    is_dag, cycles = problem.validate_dag()
    assert is_dag, f"Grafo lineal debe ser DAG válido, ciclos: {cycles}"
    print("✓ test_dag_valido_lineal")


def test_dag_valido_ramificado():
    problem = make_branched_problem()
    is_dag, cycles = problem.validate_dag()
    assert is_dag, f"Grafo ramificado debe ser DAG válido, ciclos: {cycles}"
    print("✓ test_dag_valido_ramificado")


def test_seleccion_valida_con_prerrequisitos():
    problem = make_linear_problem()
    # Selección válida: A y B (B depende de A, que está incluido)
    assert problem.is_valid_selection(["A", "B"]), "Selección [A, B] debe ser válida"
    print("✓ test_seleccion_valida_con_prerrequisitos")


def test_seleccion_invalida_sin_prerrequisito():
    problem = make_linear_problem()
    # Selección inválida: B sin A
    assert not problem.is_valid_selection(["B"]), "Selección [B] sin prerrequisito A debe ser inválida"
    print("✓ test_seleccion_invalida_sin_prerrequisito")


def test_seleccion_invalida_por_tiempo():
    problem = make_linear_problem()
    # A(10) + B(20) + C(30) = 60 > T_max=50
    assert not problem.is_valid_selection(["A", "B", "C"]), "Selección [A,B,C]=60h debe superar T_max=50"
    print("✓ test_seleccion_invalida_por_tiempo")


def test_seleccion_valida_exacta():
    problem = make_linear_problem()
    # A(10) + B(20) = 30 ≤ T_max=50
    assert problem.is_valid_selection(["A", "B"]), "Selección [A, B]=30h debe ser válida"
    print("✓ test_seleccion_valida_exacta")


def test_orden_topologico_lineal():
    problem = make_linear_problem()
    order = problem.topological_order()
    # A debe aparecer antes que B, B antes que C
    assert order.index("A") < order.index("B"), "A debe preceder a B en el orden topológico"
    assert order.index("B") < order.index("C"), "B debe preceder a C en el orden topológico"
    print("✓ test_orden_topologico_lineal")


def test_orden_topologico_ramificado():
    problem = make_branched_problem()
    order = problem.topological_order()
    assert order.index("A") < order.index("C"), "A antes que C"
    assert order.index("B") < order.index("C"), "B antes que C"
    assert order.index("C") < order.index("D"), "C antes que D"
    print("✓ test_orden_topologico_ramificado")


def test_utilidad_neutral_sin_llm():
    problem = make_linear_problem()
    # Sin evaluar por LLM, utility debe ser 5.0 (neutral)
    for course in problem.courses:
        assert course.utility == 5.0, f"Utilidad neutral esperada 5.0, got {course.utility}"
    print("✓ test_utilidad_neutral_sin_llm")


def test_valor_objetivo():
    problem = make_linear_problem()
    # Con utilidades neutras (5.0), selección de A y B → 10.0
    value = problem.objective_value(["A", "B"])
    assert value == 10.0, f"Valor objetivo esperado 10.0, got {value}"
    print("✓ test_valor_objetivo")


def test_dataset_completo_es_dag():
    """Verifica que el dataset real de 35 cursos es un DAG válido."""
    import json
    dataset_path = Path("data/raw/cursos.json")
    if not dataset_path.exists():
        print("⚠ test_dataset_completo_es_dag — dataset no encontrado, omitido")
        return

    from src.instance import load_full_problem
    problem = load_full_problem(dataset_path, t_max=9999)
    is_dag, cycles = problem.validate_dag()
    assert is_dag, f"Dataset completo debe ser DAG válido. Ciclos encontrados: {cycles}"
    print(f"✓ test_dataset_completo_es_dag ({len(problem.courses)} nodos verificados)")


# ---------------------------------------------------------------------------
# Runner manual (sin pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_dag_valido_lineal,
        test_dag_valido_ramificado,
        test_seleccion_valida_con_prerrequisitos,
        test_seleccion_invalida_sin_prerrequisito,
        test_seleccion_invalida_por_tiempo,
        test_seleccion_valida_exacta,
        test_orden_topologico_lineal,
        test_orden_topologico_ramificado,
        test_utilidad_neutral_sin_llm,
        test_valor_objetivo,
        test_dataset_completo_es_dag,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'─'*40}")
    print(f"Resultados: {passed} pasados, {failed} fallidos de {len(tests)} tests")
