"""
test_fase3.py
=============
Suite de pruebas para la Fase 3: simulación Monte Carlo, DP exacto,
Greedy y análisis de robustez.

NO requiere pytest ni ninguna dependencia externa.
Ejecutar directamente con:

    python test_fase3.py              # todos los tests
    python test_fase3.py --solo dp    # solo el grupo "dp"
    python test_fase3.py --solo mc    # solo Monte Carlo
    python test_fase3.py --solo rob   # solo robustez
    python test_fase3.py --v          # verbose (muestra detalles)

Grupos disponibles:  dp | greedy | mc | rob | integracion
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ── path ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
for _c in [ROOT, ROOT.parent, ROOT.parent.parent]:
    if (_c / "src").exists():
        ROOT = _c
        break
sys.path.insert(0, str(ROOT))

from src.problem import Course, LearningPathProblem
from src.solver.baseline import dp_knapsack_dag, greedy_by_utility_density
from src.solver.mc_sampler import mc_path_sampler, convergence_analysis
from src.solver.robustness import robustness_analysis, sensitivity_cv


# ─────────────────────────────────────────────────────────────────────────────
# Mini framework de tests
# ─────────────────────────────────────────────────────────────────────────────

VERBOSE = False

@dataclass
class _Result:
    nombre: str
    ok: bool
    msg: str = ""
    tiempo_s: float = 0.0

_resultados: List[_Result] = []


def test(nombre: str, grupo: str = "general"):
    """Decorador que registra y ejecuta un test."""
    def decorator(fn):
        def wrapper():
            t0 = time.perf_counter()
            try:
                fn()
                elapsed = time.perf_counter() - t0
                _resultados.append(_Result(f"[{grupo}] {nombre}", True, tiempo_s=elapsed))
                print(f"  ✓  {nombre:<55} ({elapsed*1000:.1f} ms)")
            except AssertionError as e:
                elapsed = time.perf_counter() - t0
                msg = str(e)
                _resultados.append(_Result(f"[{grupo}] {nombre}", False, msg, elapsed))
                print(f"  ✗  {nombre:<55} FAIL: {msg}")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                msg = f"{type(e).__name__}: {e}"
                _resultados.append(_Result(f"[{grupo}] {nombre}", False, msg, elapsed))
                print(f"  ✗  {nombre:<55} ERROR: {msg}")
                if VERBOSE:
                    traceback.print_exc()
        wrapper._grupo = grupo
        wrapper._nombre = nombre
        return wrapper
    return decorator


def afirmar(condicion: bool, msg: str = "") -> None:
    if not condicion:
        raise AssertionError(msg or "afirmación falsa")

def afirmar_aprox(a: float, b: float, msg: str = "", tol: float = 1e-6) -> None:
    if abs(a - b) > tol:
        raise AssertionError(msg or f"{a} ≠ {b} (tolerancia {tol})")

def afirmar_entre(v: float, lo: float, hi: float, msg: str = "") -> None:
    if not (lo <= v <= hi):
        raise AssertionError(msg or f"{v} no está en [{lo}, {hi}]")


# ─────────────────────────────────────────────────────────────────────────────
# Instancias de prueba sintéticas
# ─────────────────────────────────────────────────────────────────────────────

def _curso(id: str, horas: int, utilidad: int,
           prereqs: Optional[List[str]] = None) -> Course:
    c = Course(
        id=id,
        titulo=f"Curso {id}",
        descripcion=f"Descripción de {id}",
        duracion_horas=horas,
        prerrequisitos=prereqs or [],
        utilidad_relativa=utilidad,
    )
    return c


def problema_lineal() -> LearningPathProblem:
    """
    Cadena lineal: A → B → C → D
    T_max = 30 h  |  solución óptima = A+B+C (u=21, dur=25h)
    No cabe D (5h extra harían 30h exacto, pero D tiene u=3, 
    y tomar A+B+D viola prerrequisito de D que es C).
    Solución óptima real: A+B+C = u=21, dur=25h.
    """
    cursos = [
        _curso("A", 10, 9),
        _curso("B", 10, 7, ["A"]),
        _curso("C",  5, 5, ["B"]),
        _curso("D",  5, 3, ["C"]),
    ]
    return LearningPathProblem(cursos, t_max=30.0, instance_id="lineal")


def problema_bifurcacion() -> LearningPathProblem:
    """
    R → {X, Y}   (dos ramas independientes desde una raíz)
    R: 5h u=8  |  X: 10h u=9  |  Y: 15h u=6
    T_max = 20h
    Opciones: R+X (15h, u=17) vs R+Y (20h, u=14)
    Óptimo: R+X (u=17)
    """
    cursos = [
        _curso("R",  5, 8),
        _curso("X", 10, 9, ["R"]),
        _curso("Y", 15, 6, ["R"]),
    ]
    return LearningPathProblem(cursos, t_max=20.0, instance_id="bifurcacion")


def problema_sin_restriccion() -> LearningPathProblem:
    """T_max muy grande — todos los cursos caben. Solución óptima = todos."""
    cursos = [
        _curso("P", 10, 8),
        _curso("Q", 20, 6, ["P"]),
        _curso("S", 10, 9, ["P"]),
    ]
    return LearningPathProblem(cursos, t_max=1000.0, instance_id="sin_restriccion")


def problema_presupuesto_cero() -> LearningPathProblem:
    """T_max = 0 — no cabe ningún curso. Solución = vacía."""
    cursos = [_curso("A", 10, 9), _curso("B", 5, 7)]
    return LearningPathProblem(cursos, t_max=0.0, instance_id="budget_cero")


def problema_grande(n: int = 20, seed: int = 7) -> LearningPathProblem:
    """
    Instancia sintética de N cursos con estructura de DAG aleatoria.
    Útil para medir tiempos y robustez del MC.
    """
    rng = random.Random(seed)
    cursos = []
    for i in range(n):
        dur   = rng.randint(5, 30)
        util  = rng.randint(1, 10)
        # Prerrequisito: con prob 0.4, depende de un nodo anterior aleatorio
        prereqs = []
        if i > 0 and rng.random() < 0.4:
            prereqs = [f"C{rng.randint(0, i-1):02d}"]
        cursos.append(_curso(f"C{i:02d}", dur, util, prereqs))
    t_max = sum(c.duracion_horas for c in cursos) * 0.6
    return LearningPathProblem(cursos, t_max=t_max, instance_id=f"grande_{n}")


# ─────────────────────────────────────────────────────────────────────────────
# GRUPO: DP exacto
# ─────────────────────────────────────────────────────────────────────────────

@test("DP: cadena lineal — solución óptima conocida", grupo="dp")
def test_dp_lineal():
    p = problema_lineal()
    r = dp_knapsack_dag(p)
    # A+B+C = 25h, u=21 | A+B+C+D = 30h, u=24 — pero T_max=30, así que caben los 4
    afirmar(r.objective_value >= 21.0,
            f"DP debería encontrar al menos u=21, encontró {r.objective_value}")
    afirmar(r.total_duration <= p.t_max,
            f"Duración {r.total_duration} > T_max {p.t_max}")
    afirmar(p.is_valid_selection(r.selected_ids),
            "La selección del DP no es válida")
    if VERBOSE:
        print(f"\n    Selección: {r.selected_ids} | u={r.objective_value} | dur={r.total_duration}h")


@test("DP: bifurcación — elige la rama correcta", grupo="dp")
def test_dp_bifurcacion():
    p = problema_bifurcacion()
    r = dp_knapsack_dag(p)
    afirmar("R" in r.selected_ids, "Debe incluir la raíz R")
    afirmar("X" in r.selected_ids, f"Debe elegir X (u=9) sobre Y (u=6), seleccionó: {r.selected_ids}")
    afirmar("Y" not in r.selected_ids, "No debería incluir Y (inferior a X)")
    afirmar_aprox(r.objective_value, 17.0, msg=f"u esperada=17, obtenida={r.objective_value}")


@test("DP: sin restricción — selecciona todos los cursos", grupo="dp")
def test_dp_sin_restriccion():
    p = problema_sin_restriccion()
    r = dp_knapsack_dag(p)
    afirmar(len(r.selected_ids) == len(p.courses),
            f"Deben seleccionarse {len(p.courses)} cursos, seleccionó {len(r.selected_ids)}")


@test("DP: presupuesto cero — devuelve selección vacía", grupo="dp")
def test_dp_presupuesto_cero():
    p = problema_presupuesto_cero()
    r = dp_knapsack_dag(p)
    afirmar(len(r.selected_ids) == 0,
            f"Con T_max=0 no debe seleccionarse nada, seleccionó: {r.selected_ids}")
    afirmar_aprox(r.objective_value, 0.0)


@test("DP: prerrequisitos siempre satisfechos", grupo="dp")
def test_dp_prerrequisitos():
    p = problema_grande(n=15, seed=99)
    r = dp_knapsack_dag(p)
    sel = set(r.selected_ids)
    for cid in r.selected_ids:
        course = p.get_course(cid)
        for prereq in course.prerrequisitos:
            if prereq in {c.id for c in p.courses}:
                afirmar(prereq in sel,
                        f"Prerrequisito {prereq} de {cid} no está en la selección")
    if VERBOSE:
        print(f"\n    Selección: {r.selected_ids}")


@test("DP: no supera T_max en instancia grande", grupo="dp")
def test_dp_no_supera_tmax():
    p = problema_grande(n=18, seed=42)
    r = dp_knapsack_dag(p)
    afirmar(r.total_duration <= p.t_max + 1e-6,
            f"Duración {r.total_duration:.1f} > T_max {p.t_max:.1f}")


@test("DP: granularidad=5 produce resultado subóptimo pero válido", grupo="dp")
def test_dp_granularidad():
    p = problema_bifurcacion()
    r_exact = dp_knapsack_dag(p, granularidad=1)
    r_coarse = dp_knapsack_dag(p, granularidad=5)
    afirmar(p.is_valid_selection(r_coarse.selected_ids), "Resultado con granularidad=5 no es válido")
    # La solución gruesa puede ser peor o igual, pero no mejor
    afirmar(r_coarse.objective_value <= r_exact.objective_value + 1e-6,
            f"granularidad=5 no puede ser mejor que granularidad=1")


@test("DP: DPResult.summary() no lanza excepción", grupo="dp")
def test_dp_summary():
    p = problema_bifurcacion()
    r = dp_knapsack_dag(p)
    s = r.summary()
    afirmar(isinstance(s, str) and len(s) > 0)
    if VERBOSE:
        print(f"\n{s}")


# ─────────────────────────────────────────────────────────────────────────────
# GRUPO: Greedy
# ─────────────────────────────────────────────────────────────────────────────

@test("Greedy: encuentra solución no vacía", grupo="greedy")
def test_greedy_no_vacia():
    p = problema_bifurcacion()
    r = greedy_by_utility_density(p)
    afirmar(len(r.selected_ids) > 0, "Greedy devolvió selección vacía")
    afirmar(p.is_valid_selection(r.selected_ids), "Selección greedy no es válida")


@test("Greedy: no supera T_max", grupo="greedy")
def test_greedy_no_supera_tmax():
    p = problema_grande(n=20, seed=5)
    r = greedy_by_utility_density(p)
    afirmar(r.total_duration <= p.t_max + 1e-6,
            f"Greedy supera T_max: {r.total_duration:.1f} > {p.t_max:.1f}")


@test("Greedy: prerrequisitos siempre satisfechos", grupo="greedy")
def test_greedy_prerrequisitos():
    p = problema_grande(n=20, seed=13)
    r = greedy_by_utility_density(p)
    sel = set(r.selected_ids)
    for cid in r.selected_ids:
        c = p.get_course(cid)
        for prereq in c.prerrequisitos:
            if prereq in {x.id for x in p.courses}:
                afirmar(prereq in sel,
                        f"Prerrequisito {prereq} de {cid} no está seleccionado")


@test("Greedy: presupuesto cero → vacío", grupo="greedy")
def test_greedy_presupuesto_cero():
    p = problema_presupuesto_cero()
    r = greedy_by_utility_density(p)
    afirmar(len(r.selected_ids) == 0)


@test("Greedy: utilidad ≤ DP exacto (lower bound)", grupo="greedy")
def test_greedy_utilidad_le_dp():
    p = problema_bifurcacion()
    r_dp     = dp_knapsack_dag(p)
    r_greedy = greedy_by_utility_density(p)
    afirmar(r_greedy.objective_value <= r_dp.objective_value + 1e-6,
            f"Greedy ({r_greedy.objective_value}) supera DP ({r_dp.objective_value}): imposible")


# ─────────────────────────────────────────────────────────────────────────────
# GRUPO: Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

@test("MC: con semilla fija produce resultado reproducible", grupo="mc")
def test_mc_reproducible():
    p = problema_grande(n=15, seed=42)
    r1 = mc_path_sampler(p, n_iterations=200, seed=7)
    r2 = mc_path_sampler(p, n_iterations=200, seed=7)
    afirmar(r1.selected_ids == r2.selected_ids,
            f"Resultados distintos con misma semilla: {r1.selected_ids} vs {r2.selected_ids}")
    afirmar_aprox(r1.objective_value, r2.objective_value)


@test("MC: no supera T_max", grupo="mc")
def test_mc_no_supera_tmax():
    p = problema_grande(n=20, seed=1)
    r = mc_path_sampler(p, n_iterations=500, seed=42)
    afirmar(r.total_duration <= p.t_max + 1e-6,
            f"MC supera T_max: {r.total_duration:.1f} > {p.t_max:.1f}")


@test("MC: prerrequisitos siempre satisfechos", grupo="mc")
def test_mc_prerrequisitos():
    p = problema_grande(n=20, seed=2)
    r = mc_path_sampler(p, n_iterations=300, seed=42)
    sel = set(r.selected_ids)
    for cid in r.selected_ids:
        c = p.get_course(cid)
        for prereq in c.prerrequisitos:
            if prereq in {x.id for x in p.courses}:
                afirmar(prereq in sel,
                        f"Prerrequisito {prereq} de {cid} ausente en selección MC")


@test("MC: más iteraciones → misma o mejor utilidad (tendencia)", grupo="mc")
def test_mc_mas_iteraciones_mejora():
    p = problema_grande(n=15, seed=10)
    r_poco  = mc_path_sampler(p, n_iterations=50,   seed=42)
    r_medio = mc_path_sampler(p, n_iterations=500,  seed=42)
    r_mucho = mc_path_sampler(p, n_iterations=2000, seed=42)
    # No garantía estricta por semilla, pero la tendencia debe ser no decrecer
    afirmar(r_mucho.objective_value >= r_poco.objective_value - 0.1,
            f"Con más iter se espera mejor o igual resultado: "
            f"50→{r_poco.objective_value}, 2000→{r_mucho.objective_value}")
    if VERBOSE:
        print(f"\n    50 iter: u={r_poco.objective_value} | "
              f"500: u={r_medio.objective_value} | 2000: u={r_mucho.objective_value}")


@test("MC: temperatura baja (0.1) tiende a ser más greedy", grupo="mc")
def test_mc_temperatura_baja_mas_greedy():
    p = problema_grande(n=15, seed=8)
    resultados_fria  = [mc_path_sampler(p, 300, seed=s, temperature=0.1).objective_value
                        for s in range(5)]
    resultados_calida = [mc_path_sampler(p, 300, seed=s, temperature=5.0).objective_value
                         for s in range(5)]
    std_fria  = (sum((x - sum(resultados_fria)/5)**2  for x in resultados_fria)  / 5) ** 0.5
    std_calida = (sum((x - sum(resultados_calida)/5)**2 for x in resultados_calida) / 5) ** 0.5
    # T baja debe producir menos varianza (soluciones más consistentes)
    afirmar(std_fria <= std_calida + 1.0,
            f"std temperatura=0.1 ({std_fria:.2f}) debería ≤ std temperatura=5.0 ({std_calida:.2f})")
    if VERBOSE:
        print(f"\n    std T=0.1: {std_fria:.3f} | std T=5.0: {std_calida:.3f}")


@test("MC: convergence_analysis devuelve checkpoints completos", grupo="mc")
def test_mc_convergence_analysis():
    p = problema_grande(n=12, seed=3)
    checkpoints = [50, 100, 200, 500]
    curva = convergence_analysis(p, n_iterations=500, checkpoints=checkpoints, seed=42)
    for ck in checkpoints:
        afirmar(ck in curva, f"Checkpoint {ck} ausente en resultado")
        afirmar(curva[ck] >= 0, f"Utilidad negativa en checkpoint {ck}")
    # La curva debe ser monótonamente no decreciente
    vals = [curva[ck] for ck in checkpoints]
    for i in range(len(vals)-1):
        afirmar(vals[i+1] >= vals[i] - 1e-9,
                f"Curva decreció en checkpoint {checkpoints[i+1]}: "
                f"{vals[i]:.1f} → {vals[i+1]:.1f}")
    if VERBOSE:
        print(f"\n    Curva: {dict(zip(checkpoints, vals))}")


@test("MC: sin utilidades asignadas → ValueError", grupo="mc")
def test_mc_sin_utilidades():
    cursos = [
        Course("A", "A", "", 10, [], utilidad_relativa=None),
        Course("B", "B", "", 10, ["A"], utilidad_relativa=None),
    ]
    p = LearningPathProblem(cursos, t_max=20.0)
    try:
        mc_path_sampler(p, n_iterations=10)
        afirmar(False, "Debería lanzar ValueError")
    except ValueError:
        pass  # esperado


@test("MC: presupuesto cero → selección vacía", grupo="mc")
def test_mc_presupuesto_cero():
    p = problema_presupuesto_cero()
    r = mc_path_sampler(p, n_iterations=100, seed=42)
    afirmar(len(r.selected_ids) == 0,
            f"Con T_max=0 no debe seleccionarse nada: {r.selected_ids}")


@test("MC: MCResult.summary() no lanza excepción", grupo="mc")
def test_mc_summary():
    p = problema_bifurcacion()
    r = mc_path_sampler(p, n_iterations=100, seed=42)
    s = r.summary()
    afirmar(isinstance(s, str) and len(s) > 0)
    if VERBOSE:
        print(f"\n{s}")


@test("MC: n_feasible ≤ n_iterations", grupo="mc")
def test_mc_n_feasible():
    p = problema_grande(n=15, seed=6)
    r = mc_path_sampler(p, n_iterations=300, seed=42)
    afirmar(0 <= r.n_feasible <= r.n_iterations,
            f"n_feasible={r.n_feasible} inválido para n_iter={r.n_iterations}")


@test("MC: convergence_iter está dentro del rango de iteraciones", grupo="mc")
def test_mc_convergence_iter():
    p = problema_grande(n=12, seed=4)
    r = mc_path_sampler(p, n_iterations=500, seed=42)
    afirmar(1 <= r.convergence_iter <= 500,
            f"convergence_iter={r.convergence_iter} fuera de [1, 500]")


# ─────────────────────────────────────────────────────────────────────────────
# GRUPO: Robustez
# ─────────────────────────────────────────────────────────────────────────────

@test("Rob: P(factible) ∈ [0, 1]", grupo="rob")
def test_rob_p_feasible_range():
    p = problema_grande(n=12, seed=5)
    r = dp_knapsack_dag(p)
    rob = robustness_analysis(p, r.selected_ids, n_simulations=1000, cv=0.20, seed=42)
    afirmar_entre(rob.p_feasible, 0.0, 1.0,
                  f"p_feasible={rob.p_feasible} fuera de [0,1]")


@test("Rob: IC 95% contiene p_feasible", grupo="rob")
def test_rob_ic_contains_mean():
    p = problema_grande(n=12, seed=5)
    r = dp_knapsack_dag(p)
    rob = robustness_analysis(p, r.selected_ids, n_simulations=2000, cv=0.20, seed=42)
    afirmar(rob.ci_95_lower <= rob.p_feasible <= rob.ci_95_upper,
            f"p_feasible={rob.p_feasible:.3f} fuera del IC "
            f"[{rob.ci_95_lower:.3f}, {rob.ci_95_upper:.3f}]")


@test("Rob: ruta con gran margen tiene P(factible) alta", grupo="rob")
def test_rob_gran_margen_alta_prob():
    # Solo un curso de 10h, T_max=100h → margen del 90%
    p = LearningPathProblem(
        [_curso("A", 10, 9)],
        t_max=100.0,
        instance_id="margen_grande"
    )
    rob = robustness_analysis(p, ["A"], n_simulations=2000, cv=0.30, seed=42)
    afirmar(rob.p_feasible >= 0.99,
            f"Con 90% de margen P(fact) debe ser ≥ 99%, obtuvo {rob.p_feasible:.2%}")


@test("Rob: ruta ajustada tiene P(factible) < ruta holgada", grupo="rob")
def test_rob_ruta_ajustada_menor_prob():
    p = problema_grande(n=15, seed=7)
    r_dp = dp_knapsack_dag(p)

    # Ruta holgada: solo el primer curso del DP
    ruta_holgada = r_dp.selected_ids[:1]
    # Ruta ajustada: todos los cursos del DP (más cerca de T_max)
    ruta_ajustada = r_dp.selected_ids

    dur_h = p.selection_duration(ruta_holgada)
    dur_a = p.selection_duration(ruta_ajustada)

    if dur_h >= dur_a:
        # Casos degenerados donde el DP solo selecciona 1 curso
        return

    rob_h = robustness_analysis(p, ruta_holgada, n_simulations=2000, cv=0.25, seed=42)
    rob_a = robustness_analysis(p, ruta_ajustada, n_simulations=2000, cv=0.25, seed=42)

    afirmar(rob_h.p_feasible >= rob_a.p_feasible,
            f"Ruta holgada ({dur_h:.0f}h) P={rob_h.p_feasible:.2%} debería ≥ "
            f"ruta ajustada ({dur_a:.0f}h) P={rob_a.p_feasible:.2%}")
    if VERBOSE:
        print(f"\n    Holgada:  {dur_h:.0f}h → P={rob_h.p_feasible:.2%}")
        print(f"    Ajustada: {dur_a:.0f}h → P={rob_a.p_feasible:.2%}")


@test("Rob: cv=0 → duración simulada = duración nominal", grupo="rob")
def test_rob_cv_zero_duration_exact():
    p = problema_bifurcacion()
    r = dp_knapsack_dag(p)
    rob = robustness_analysis(p, r.selected_ids, n_simulations=500, cv=0.0, seed=42)
    # CORREGIDO: el mensaje ahora se pasa antes de `tol`
    afirmar_aprox(rob.mean_duration, r.total_duration,
                  f"cv=0: media simulada {rob.mean_duration:.2f} ≠ nominal {r.total_duration:.2f}",
                  tol=1e-3)


@test("Rob: cv mayor → std_duration mayor", grupo="rob")
def test_rob_cv_mayor_std_mayor():
    p = problema_grande(n=12, seed=3)
    r = dp_knapsack_dag(p)
    rob_01 = robustness_analysis(p, r.selected_ids, n_simulations=2000, cv=0.10, seed=42)
    rob_03 = robustness_analysis(p, r.selected_ids, n_simulations=2000, cv=0.30, seed=42)
    afirmar(rob_03.std_duration > rob_01.std_duration,
            f"cv=0.30 debería tener mayor std: {rob_03.std_duration:.2f} vs {rob_01.std_duration:.2f}")


@test("Rob: percentil_95 ≥ media simulada", grupo="rob")
def test_rob_percentil_95_ge_mean():
    p = problema_grande(n=12, seed=9)
    r = dp_knapsack_dag(p)
    rob = robustness_analysis(p, r.selected_ids, n_simulations=2000, cv=0.20, seed=42)
    afirmar(rob.percentile_95_dur >= rob.mean_duration - 1e-6,
            f"P95={rob.percentile_95_dur:.1f} < media={rob.mean_duration:.1f}")


@test("Rob: ID inválido → ValueError", grupo="rob")
def test_rob_id_invalido():
    p = problema_bifurcacion()
    try:
        robustness_analysis(p, ["INEXISTENTE"], n_simulations=100, cv=0.20)
        afirmar(False, "Debería lanzar ValueError")
    except ValueError:
        pass  # esperado


@test("Rob: sensitivity_cv devuelve un valor por cada cv", grupo="rob")
def test_rob_sensitivity_cv():
    p = problema_bifurcacion()
    r = dp_knapsack_dag(p)
    cv_vals = [0.10, 0.20, 0.30]
    sv = sensitivity_cv(p, r.selected_ids, cv_values=cv_vals, n_simulations=500, seed=42)
    afirmar(len(sv) == len(cv_vals),
            f"sensitivity_cv devolvió {len(sv)} valores, esperados {len(cv_vals)}")
    for cv, pf in sv.items():
        afirmar_entre(pf, 0.0, 1.0, f"P(fact) fuera de [0,1] para cv={cv}")
    if VERBOSE:
        print(f"\n    {sv}")


@test("Rob: RobustnessResult.summary() no lanza excepción", grupo="rob")
def test_rob_summary():
    p = problema_bifurcacion()
    r = dp_knapsack_dag(p)
    rob = robustness_analysis(p, r.selected_ids, n_simulations=500, cv=0.20, seed=42)
    s = rob.summary()
    afirmar(isinstance(s, str) and len(s) > 0)
    if VERBOSE:
        print(f"\n{s}")


@test("Rob: risk_level es uno de los tres valores esperados", grupo="rob")
def test_rob_risk_level():
    p = problema_grande(n=12, seed=2)
    r = dp_knapsack_dag(p)
    rob = robustness_analysis(p, r.selected_ids, n_simulations=500, cv=0.20, seed=42)
    afirmar(any(k in rob.risk_level for k in ("BAJO", "MEDIO", "ALTO")),
            f"risk_level inesperado: {rob.risk_level!r}")


# ─────────────────────────────────────────────────────────────────────────────
# GRUPO: Integración (los tres solvers juntos)
# ─────────────────────────────────────────────────────────────────────────────

@test("Integración: DP ≥ Greedy ≥ MC no siempre, pero DP es óptimo", grupo="integracion")
def test_integracion_dp_ge_greedy_ge_mc():
    """El DP es globalmente óptimo: ningún otro solver puede superarlo."""
    for seed in [1, 2, 3, 5, 8]:
        p   = problema_grande(n=12, seed=seed)
        dp  = dp_knapsack_dag(p)
        grd = greedy_by_utility_density(p)
        mc  = mc_path_sampler(p, n_iterations=1000, seed=42)

        afirmar(dp.objective_value >= grd.objective_value - 1e-6,
                f"seed={seed}: DP ({dp.objective_value}) < Greedy ({grd.objective_value})")
        afirmar(dp.objective_value >= mc.objective_value - 1e-6,
                f"seed={seed}: DP ({dp.objective_value}) < MC ({mc.objective_value})")


@test("Integración: los tres solvers producen selecciones válidas", grupo="integracion")
def test_integracion_selecciones_validas():
    p   = problema_grande(n=15, seed=99)
    dp  = dp_knapsack_dag(p)
    grd = greedy_by_utility_density(p)
    mc  = mc_path_sampler(p, n_iterations=500, seed=42)

    afirmar(p.is_valid_selection(dp.selected_ids),  "DP: selección inválida")
    afirmar(p.is_valid_selection(grd.selected_ids), "Greedy: selección inválida")
    afirmar(p.is_valid_selection(mc.selected_ids),  "MC: selección inválida")


@test("Integración: MC con 5000 iter se acerca al DP en instancia mediana", grupo="integracion")
def test_integracion_mc_se_aproxima_dp():
    p   = problema_grande(n=18, seed=55)
    dp  = dp_knapsack_dag(p)
    mc  = mc_path_sampler(p, n_iterations=5000, seed=42)

    gap_pct = (dp.objective_value - mc.objective_value) / max(dp.objective_value, 1) * 100
    afirmar(gap_pct <= 20.0,
            f"MC demasiado lejos del DP: gap={gap_pct:.1f}%")
    if VERBOSE:
        print(f"\n    DP={dp.objective_value:.1f} | MC={mc.objective_value:.1f} | gap={gap_pct:.1f}%")


@test("Integración: robustez sobre solución DP y MC (ambas válidas)", grupo="integracion")
def test_integracion_robustez_dp_mc():
    p   = problema_grande(n=12, seed=11)
    dp  = dp_knapsack_dag(p)
    mc  = mc_path_sampler(p, n_iterations=500, seed=42)

    if not dp.selected_ids or not mc.selected_ids:
        return  # nada que analizar si alguno es vacío

    rob_dp = robustness_analysis(p, dp.selected_ids, n_simulations=1000, cv=0.20, seed=42)
    rob_mc = robustness_analysis(p, mc.selected_ids, n_simulations=1000, cv=0.20, seed=42)

    afirmar_entre(rob_dp.p_feasible, 0.0, 1.0)
    afirmar_entre(rob_mc.p_feasible, 0.0, 1.0)


@test("Integración: tiempo de ejecución razonable para n=20", grupo="integracion")
def test_integracion_tiempo_ejecucion():
    p = problema_grande(n=20, seed=77)
    t0 = time.perf_counter()
    dp_knapsack_dag(p)
    t_dp = time.perf_counter() - t0

    t0 = time.perf_counter()
    mc_path_sampler(p, n_iterations=1000, seed=42)
    t_mc = time.perf_counter() - t0

    afirmar(t_dp < 5.0,  f"DP demasiado lento: {t_dp:.2f}s (esperado < 5s)")
    afirmar(t_mc < 10.0, f"MC demasiado lento: {t_mc:.2f}s (esperado < 10s)")
    if VERBOSE:
        print(f"\n    DP: {t_dp:.3f}s | MC(1000iter): {t_mc:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

GRUPOS = {
    "dp":           [v for k, v in globals().items() if callable(v) and getattr(v, "_grupo", "") == "dp"],
    "greedy":       [v for k, v in globals().items() if callable(v) and getattr(v, "_grupo", "") == "greedy"],
    "mc":           [v for k, v in globals().items() if callable(v) and getattr(v, "_grupo", "") == "mc"],
    "rob":          [v for k, v in globals().items() if callable(v) and getattr(v, "_grupo", "") == "rob"],
    "integracion":  [v for k, v in globals().items() if callable(v) and getattr(v, "_grupo", "") == "integracion"],
}


def _print_resumen() -> None:
    total   = len(_resultados)
    ok      = sum(1 for r in _resultados if r.ok)
    fallidos = [r for r in _resultados if not r.ok]

    print(f"\n{'═'*64}")
    print(f"  RESUMEN: {ok}/{total} tests pasaron")
    if fallidos:
        print(f"\n  Tests fallidos ({len(fallidos)}):")
        for r in fallidos:
            print(f"    ✗  {r.nombre}")
            print(f"       {r.msg}")
    t_total = sum(r.tiempo_s for r in _resultados)
    print(f"\n  Tiempo total: {t_total*1000:.0f} ms")
    print(f"{'═'*64}")


def main() -> int:
    global VERBOSE

    parser = argparse.ArgumentParser(description="Tests de la Fase 3.")
    parser.add_argument("--solo",  help=f"Ejecutar solo este grupo: {list(GRUPOS)}")
    parser.add_argument("--v", action="store_true", help="Verbose.")
    args = parser.parse_args()
    VERBOSE = args.v

    grupos_a_correr = {args.solo: GRUPOS[args.solo]} if args.solo else GRUPOS

    if args.solo and args.solo not in GRUPOS:
        print(f"Grupo '{args.solo}' no existe. Disponibles: {list(GRUPOS)}")
        return 1

    for nombre_grupo, tests in grupos_a_correr.items():
        print(f"\n{'─'*64}")
        print(f"  Grupo: {nombre_grupo.upper()}  ({len(tests)} tests)")
        print(f"{'─'*64}")
        for t in tests:
            t()

    _print_resumen()
    return 0 if all(r.ok for r in _resultados) else 1


if __name__ == "__main__":
    sys.exit(main())