"""
test_simulacion.py  (tests/test_simulacion.py)
===============================================
Pruebas unitarias para los módulos de simulación Monte Carlo.

Todos los tests son independientes de la API del LLM: usan un problema
sintético pequeño con utilidades fijas.

Ejecutar:
    python tests/test_simulacion.py
    # o con pytest:
    python -m pytest tests/test_simulacion.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver.mc_sampler import mc_path_sampler, convergence_analysis, MCResult
from src.solver.robustness import robustness_analysis, sensitivity_cv, RobustnessResult


# ---------------------------------------------------------------------------
# Problema sintético para tests (sin importar problem.py para aislamiento)
# ---------------------------------------------------------------------------

class _FakeCourse:
    """Simula un objeto Course con los campos que usa el solver."""
    def __init__(self, id, duration, utility, prereqs=None):
        self.id = id
        self.duracion_horas = duration
        self.prerrequisitos = prereqs or []
        self.utilidad_relativa = int(utility)

    @property
    def duration(self):
        return float(self.duracion_horas)

    @property
    def utility(self):
        return float(self.utilidad_relativa)


class _FakeProblem:
    """Simula un LearningPathProblem con los campos que usa el solver."""
    def __init__(self, courses, t_max, instance_id="test"):
        self.courses = courses
        self.t_max = t_max
        self.instance_id = instance_id


def make_linear_problem():
    """
    Problema lineal simple: A → B → C
    A: d=10, u=3 | B: d=20, u=7 | C: d=30, u=9
    T_max = 50 h → solo A+B (30h) es factible; C+A+B = 60h > 50
    """
    return _FakeProblem(
        courses=[
            _FakeCourse("A", 10, 3, []),
            _FakeCourse("B", 20, 7, ["A"]),
            _FakeCourse("C", 30, 9, ["B"]),
        ],
        t_max=50,
    )


def make_branched_problem():
    """
    Problema con ramificación:
        A(d=15, u=5), B(d=10, u=4) → C(d=20, u=8) → D(d=25, u=9)
    T_max = 80h
    Ruta óptima esperada: A+B+C+D = 70h, u=26
    """
    return _FakeProblem(
        courses=[
            _FakeCourse("A", 15, 5, []),
            _FakeCourse("B", 10, 4, []),
            _FakeCourse("C", 20, 8, ["A", "B"]),
            _FakeCourse("D", 25, 9, ["C"]),
        ],
        t_max=80,
    )


# ---------------------------------------------------------------------------
# Tests: mc_path_sampler
# ---------------------------------------------------------------------------

def test_mc_devuelve_mcresult():
    problem = make_linear_problem()
    result = mc_path_sampler(problem, n_iterations=100, seed=42)
    assert isinstance(result, MCResult)
    print("✓ test_mc_devuelve_mcresult")


def test_mc_respeta_t_max():
    problem = make_linear_problem()
    result = mc_path_sampler(problem, n_iterations=200, seed=42)
    total = sum(
        next(c.duration for c in problem.courses if c.id == cid)
        for cid in result.selected_ids
    )
    assert total <= problem.t_max, f"Duración {total}h supera T_max={problem.t_max}h"
    print(f"✓ test_mc_respeta_t_max ({total:.0f}h ≤ {problem.t_max:.0f}h)")


def test_mc_respeta_prerrequisitos():
    """Ningún curso seleccionado puede carecer de sus prerrequisitos."""
    problem = make_branched_problem()
    result = mc_path_sampler(problem, n_iterations=500, seed=0)
    selected_set = set(result.selected_ids)
    prereq_map = {c.id: c.prerrequisitos for c in problem.courses}

    for cid in selected_set:
        for p in prereq_map[cid]:
            assert p in selected_set, (
                f"Prerrequisito violado: {cid} incluido pero {p} no está en la selección"
            )
    print("✓ test_mc_respeta_prerrequisitos")


def test_mc_encuentra_ruta_no_vacia():
    """Con T_max holgado, el sampler debe encontrar al menos un curso."""
    problem = make_branched_problem()
    result = mc_path_sampler(problem, n_iterations=50, seed=7)
    assert len(result.selected_ids) > 0, "La ruta no debe estar vacía con T_max holgado"
    print(f"✓ test_mc_encuentra_ruta_no_vacia ({len(result.selected_ids)} cursos)")


def test_mc_reproducibilidad():
    """Dos ejecuciones con la misma semilla deben dar el mismo resultado."""
    problem = make_branched_problem()
    r1 = mc_path_sampler(problem, n_iterations=200, seed=123)
    r2 = mc_path_sampler(problem, n_iterations=200, seed=123)
    assert r1.selected_ids == r2.selected_ids
    assert r1.objective_value == r2.objective_value
    print("✓ test_mc_reproducibilidad")


def test_mc_mas_iteraciones_no_empeora():
    """Con más iteraciones la utilidad no puede ser menor (resultado acumulativo)."""
    problem = make_branched_problem()
    r100 = mc_path_sampler(problem, n_iterations=100, seed=1)
    r500 = mc_path_sampler(problem, n_iterations=500, seed=1)
    assert r500.objective_value >= r100.objective_value, (
        f"500 iters (u={r500.objective_value}) peor que 100 iters (u={r100.objective_value})"
    )
    print(
        f"✓ test_mc_mas_iteraciones_no_empeora "
        f"(100→u={r100.objective_value:.1f}, 500→u={r500.objective_value:.1f})"
    )


def test_mc_utilidad_por_hora():
    """utilidad_por_hora debe ser objective_value / total_duration."""
    problem = make_branched_problem()
    result = mc_path_sampler(problem, n_iterations=200, seed=42)
    if result.total_duration > 0:
        expected = result.objective_value / result.total_duration
        assert abs(result.utilidad_por_hora - expected) < 1e-6
    print(f"✓ test_mc_utilidad_por_hora ({result.utilidad_por_hora:.3f} u/h)")


def test_mc_falla_sin_utilidades():
    """Debe lanzar ValueError si no hay utilidades asignadas."""
    courses = [_FakeCourse("X", 20, 5, [])]
    courses[0].utilidad_relativa = None  # simular sin evaluación LLM
    problem = _FakeProblem(courses, t_max=50)

    try:
        mc_path_sampler(problem, n_iterations=10)
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass
    print("✓ test_mc_falla_sin_utilidades")


def test_convergence_analysis_checkpoints():
    """convergence_analysis debe devolver una clave por cada checkpoint."""
    problem = make_branched_problem()
    checkpoints = [10, 50, 100]
    results = convergence_analysis(problem, n_iterations=100, checkpoints=checkpoints, seed=0)
    assert set(results.keys()) == set(checkpoints), f"Checkpoints incorrectos: {results.keys()}"
    print(f"✓ test_convergence_analysis_checkpoints: {results}")


def test_convergence_es_monotona():
    """La mejor utilidad acumulada debe ser no decreciente."""
    problem = make_branched_problem()
    results = convergence_analysis(problem, n_iterations=500, seed=5)
    valores = list(results.values())
    for i in range(len(valores) - 1):
        assert valores[i] <= valores[i + 1], (
            f"No monótona: checkpoint {i} ({valores[i]}) > checkpoint {i+1} ({valores[i+1]})"
        )
    print(f"✓ test_convergence_es_monotona: {results}")


# ---------------------------------------------------------------------------
# Tests: robustness_analysis
# ---------------------------------------------------------------------------

def test_robustez_devuelve_resultado():
    problem = make_branched_problem()
    result = robustness_analysis(problem, ["A", "B", "C"], n_simulations=500, cv=0.20, seed=42)
    assert isinstance(result, RobustnessResult)
    print("✓ test_robustez_devuelve_resultado")


def test_robustez_probabilidad_en_rango():
    problem = make_branched_problem()
    result = robustness_analysis(problem, ["A", "B", "C"], n_simulations=1000, cv=0.20, seed=0)
    assert 0.0 <= result.p_feasible <= 1.0, f"p_feasible fuera de [0,1]: {result.p_feasible}"
    print(f"✓ test_robustez_probabilidad_en_rango (p={result.p_feasible:.2%})")


def test_robustez_ruta_holgada_alta_probabilidad():
    """Si la ruta usa poco tiempo, la probabilidad de ser factible debe ser alta."""
    problem = make_branched_problem()
    # Solo A+B: d=25h << T_max=80h → casi siempre factible incluso con cv alto
    result = robustness_analysis(problem, ["A", "B"], n_simulations=2000, cv=0.30, seed=1)
    assert result.p_feasible > 0.98, (
        f"Ruta holgada debería tener p > 98%, got {result.p_feasible:.2%}"
    )
    print(f"✓ test_robustez_ruta_holgada_alta_probabilidad (p={result.p_feasible:.2%})")


def test_robustez_ruta_ajustada_menor_probabilidad():
    """Si la ruta usa casi todo el presupuesto, la probabilidad baja."""
    problem = make_branched_problem()
    # A+B+C+D: d=70h vs T_max=80h → con cv=0.30 hay riesgo real
    result_ajustada = robustness_analysis(
        problem, ["A", "B", "C", "D"], n_simulations=2000, cv=0.30, seed=2
    )
    result_holgada = robustness_analysis(
        problem, ["A", "B"], n_simulations=2000, cv=0.30, seed=2
    )
    assert result_ajustada.p_feasible < result_holgada.p_feasible, (
        "Ruta ajustada debe tener menor p que ruta holgada"
    )
    print(
        f"✓ test_robustez_ruta_ajustada_menor_probabilidad "
        f"(ajustada={result_ajustada.p_feasible:.2%} < holgada={result_holgada.p_feasible:.2%})"
    )


def test_robustez_cv_cero_determinista():
    """Con cv=0 no hay variabilidad: p_feasible debe ser exactamente 0% o 100%."""
    problem = make_branched_problem()
    # A+B+C: d=45h < T_max=80h → factible con probabilidad 1.0 cuando cv=0
    result = robustness_analysis(problem, ["A", "B", "C"], n_simulations=500, cv=0.0, seed=0)
    assert result.p_feasible in (0.0, 1.0), (
        f"Con cv=0 p_feasible debe ser 0 o 1, got {result.p_feasible}"
    )
    print(f"✓ test_robustez_cv_cero_determinista (p={result.p_feasible:.0%})")


def test_robustez_ic95_incluye_p():
    """El IC 95% debe contener la estimación puntual."""
    problem = make_branched_problem()
    result = robustness_analysis(problem, ["A", "B", "C", "D"], n_simulations=1000, cv=0.20, seed=3)
    assert result.ci_95_lower <= result.p_feasible <= result.ci_95_upper, (
        f"p_feasible={result.p_feasible:.4f} fuera de IC [{result.ci_95_lower:.4f}, {result.ci_95_upper:.4f}]"
    )
    print(f"✓ test_robustez_ic95_incluye_p (IC=[{result.ci_95_lower:.2%},{result.ci_95_upper:.2%}])")


def test_robustez_id_invalido():
    problem = make_branched_problem()
    try:
        robustness_analysis(problem, ["A", "INEXISTENTE"], n_simulations=100)
        assert False, "Debe lanzar ValueError con ID inválido"
    except ValueError:
        pass
    print("✓ test_robustez_id_invalido")


def test_sensibilidad_cv_es_decreciente():
    """A mayor cv (más variabilidad), P(factible) no debe aumentar."""
    problem = make_branched_problem()
    cv_vals = [0.05, 0.10, 0.20, 0.30]
    results = sensitivity_cv(problem, ["A", "B", "C", "D"], cv_vals, n_simulations=1000, seed=9)
    probs = [results[cv] for cv in cv_vals]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1] - 0.05, (
            f"P no decreciente: cv={cv_vals[i]} (p={probs[i]:.2%}) "
            f"< cv={cv_vals[i+1]} (p={probs[i+1]:.2%}) — puede pasar por aleatoriedad, tolerancia 5%"
        )
    print(f"✓ test_sensibilidad_cv_es_decreciente: {[f'{cv}→{p:.0%}' for cv, p in results.items()]}")


def test_nivel_riesgo():
    """risk_level debe clasificar correctamente según p_feasible."""
    from src.solver.robustness import RobustnessResult
    r_high = RobustnessResult([], 100, 100, 0.2, 0.95, 90, 5, 0.93, 0.97, 95)
    r_med  = RobustnessResult([], 100, 100, 0.2, 0.80, 90, 5, 0.78, 0.82, 95)
    r_low  = RobustnessResult([], 100, 100, 0.2, 0.55, 90, 5, 0.53, 0.57, 95)
    assert "BAJO"  in r_high.risk_level
    assert "MEDIO" in r_med.risk_level
    assert "ALTO"  in r_low.risk_level
    print("✓ test_nivel_riesgo")


# ---------------------------------------------------------------------------
# Runner manual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # MC Sampler
        test_mc_devuelve_mcresult,
        test_mc_respeta_t_max,
        test_mc_respeta_prerrequisitos,
        test_mc_encuentra_ruta_no_vacia,
        test_mc_reproducibilidad,
        test_mc_mas_iteraciones_no_empeora,
        test_mc_utilidad_por_hora,
        test_mc_falla_sin_utilidades,
        test_convergence_analysis_checkpoints,
        test_convergence_es_monotona,
        # Robustez
        test_robustez_devuelve_resultado,
        test_robustez_probabilidad_en_rango,
        test_robustez_ruta_holgada_alta_probabilidad,
        test_robustez_ruta_ajustada_menor_probabilidad,
        test_robustez_cv_cero_determinista,
        test_robustez_ic95_incluye_p,
        test_robustez_id_invalido,
        test_sensibilidad_cv_es_decreciente,
        test_nivel_riesgo,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"✗ {fn.__name__}: {e}")
            failed += 1

    print(f"\n{'─'*55}")
    print(f"Resultados: {passed} pasados, {failed} fallidos de {len(tests)} tests")
