"""
experimento_3_escalabilidad.py
==============================
Experimento 3 — Escalabilidad del sistema en instancias A, B y C

Hipótesis:
    - El solver DP exacto es óptimo pero su tiempo crece con n×W.
    - El Greedy es muy rápido pero sacrifica utilidad.
    - Monte Carlo escala linealmente con n_iter, independientemente del tamaño
      del problema, y aproxima bien al óptimo con suficientes iteraciones.

Diseño:
    - Se comparan tres solvers: DP exacto, Greedy por densidad, Monte Carlo.
    - Instancias: A (pequeña), B (mediana), C (grande).
    - Métricas: utilidad obtenida, tiempo de ejecución, gap respecto al DP (%).
    - Para MC se ejecutan N_REPS réplicas para obtener intervalos de confianza.
    - El DP se ejecuta una sola vez (determinista).

Salida:
    results/experimento_3_escalabilidad.json
    results/experimento_3_escalabilidad.csv
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.problem import LearningPathProblem
from src.solver.baseline import dp_knapsack_dag, greedy_by_utility_density
from src.solver.mc_sampler import mc_path_sampler

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("exp3")

# ── Parámetros ────────────────────────────────────────────────────────────────
INSTANCES = [
    ("A", ROOT / "data/processed/instancia_A_evaluada.json"),
    ("B", ROOT / "data/processed/instancia_B_evaluada.json"),
    ("C", ROOT / "data/processed/instancia_C_evaluada.json"),
]
N_ITER_MC   = 2000
TEMPERATURE = 1.0
N_REPS_MC   = 10
BASE_SEED   = 42
MAX_DP_STATES = 500_000   # mismo umbral que llm_assisted.py
RESULTS_DIR = ROOT / "results"


def _run_dp(problem: LearningPathProblem) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """Ejecuta DP y devuelve (utilidad, tiempo_s, n_estados). None si no es tratable."""
    n = len(problem.courses)
    W = int(problem.t_max)
    if n * W > MAX_DP_STATES:
        return None, None, None

    t0 = time.perf_counter()
    result = dp_knapsack_dag(problem)
    elapsed = time.perf_counter() - t0
    return result.objective_value, elapsed, result.n_states


def _run_greedy(problem: LearningPathProblem) -> tuple[float, float]:
    t0 = time.perf_counter()
    result = greedy_by_utility_density(problem)
    elapsed = time.perf_counter() - t0
    return result.objective_value, elapsed


def _run_mc(problem: LearningPathProblem, n_reps: int) -> tuple[float, float, float, float]:
    """Devuelve (u_media, u_std, t_media, t_std) de n_reps réplicas."""
    utilidades: list[float] = []
    tiempos:    list[float] = []

    for rep in range(n_reps):
        seed = BASE_SEED + rep * 97
        t0 = time.perf_counter()
        result = mc_path_sampler(
            problem,
            n_iterations=N_ITER_MC,
            temperature=TEMPERATURE,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0
        utilidades.append(result.objective_value)
        tiempos.append(elapsed)

    u_m = mean(utilidades)
    u_s = stdev(utilidades) if len(utilidades) > 1 else 0.0
    t_m = mean(tiempos)
    t_s = stdev(tiempos) if len(tiempos) > 1 else 0.0
    return u_m, u_s, t_m, t_s


def run() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENTO 3 — Escalabilidad (instancias A, B, C)")
    print(f"MC: n_iter={N_ITER_MC}, T={TEMPERATURE}, réplicas={N_REPS_MC}")
    print(f"{'='*80}")

    all_results: list[dict[str, Any]] = []

    for label, path in INSTANCES:
        if not path.exists():
            print(f"\n⚠  Instancia {label} no encontrada: {path}. Saltando.")
            continue

        problem = load_instance(path)

        if any(c.utilidad_relativa is None for c in problem.courses):
            print(f"\n⚠  Instancia {label} sin utilidades. Saltando.")
            continue

        n  = len(problem.courses)
        W  = int(problem.t_max)
        edges = sum(
            1 for c in problem.courses for p in c.prerrequisitos
            if any(x.id == p for x in problem.courses)
        )

        print(f"\n── Instancia {label}: {problem.instance_id} ──")
        print(f"   Nodos={n} | Aristas={edges} | T_max={problem.t_max:.0f}h | n×W={n*W:,}")

        # ── DP ────────────────────────────────────────────────────────────────
        dp_u, dp_t, dp_estados = _run_dp(problem)
        if dp_u is not None:
            print(f"   DP     : u={dp_u:.2f}  t={dp_t:.4f}s  estados={dp_estados:,}")
        else:
            print(f"   DP     : omitido (n×W={n*W:,} > {MAX_DP_STATES:,})")

        # ── Greedy ────────────────────────────────────────────────────────────
        gr_u, gr_t = _run_greedy(problem)
        gap_greedy = ((dp_u - gr_u) / dp_u * 100) if dp_u else None
        print(
            f"   Greedy : u={gr_u:.2f}  t={gr_t:.6f}s"
            + (f"  gap={gap_greedy:.1f}%" if gap_greedy is not None else "")
        )

        # ── Monte Carlo ────────────────────────────────────────────────────────
        mc_u_m, mc_u_s, mc_t_m, mc_t_s = _run_mc(problem, N_REPS_MC)
        gap_mc = ((dp_u - mc_u_m) / dp_u * 100) if dp_u else None
        print(
            f"   MC     : u={mc_u_m:.2f}±{mc_u_s:.3f}  t={mc_t_m:.4f}±{mc_t_s:.4f}s"
            + (f"  gap={gap_mc:.1f}%" if gap_mc is not None else "")
        )

        all_results.append({
            "instancia_label":     label,
            "instancia_id":        problem.instance_id,
            "n_cursos":            n,
            "n_aristas":           edges,
            "t_max":               problem.t_max,
            "espacio_dp":          n * W,
            # DP
            "dp_utilidad":         round(dp_u, 4) if dp_u is not None else None,
            "dp_tiempo_s":         round(dp_t, 6) if dp_t is not None else None,
            "dp_n_estados":        dp_estados,
            "dp_ejecutado":        dp_u is not None,
            # Greedy
            "greedy_utilidad":     round(gr_u, 4),
            "greedy_tiempo_s":     round(gr_t, 6),
            "greedy_gap_pct":      round(gap_greedy, 4) if gap_greedy is not None else None,
            # Monte Carlo
            "mc_utilidad_media":   round(mc_u_m, 4),
            "mc_utilidad_std":     round(mc_u_s, 4),
            "mc_tiempo_medio_s":   round(mc_t_m, 6),
            "mc_tiempo_std_s":     round(mc_t_s, 6),
            "mc_gap_pct":          round(gap_mc, 4) if gap_mc is not None else None,
            "mc_n_iter":           N_ITER_MC,
            "mc_temperatura":      TEMPERATURE,
            "mc_n_reps":           N_REPS_MC,
        })

    # ── Persistencia ──────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "experimento_3_escalabilidad.json"
    csv_path  = RESULTS_DIR / "experimento_3_escalabilidad.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experimento":  3,
                "descripcion":  "Escalabilidad DP vs Greedy vs MC en instancias A, B, C",
                "resultados":   all_results,
            },
            f, ensure_ascii=False, indent=2,
        )

    if all_results:
        headers = list(all_results[0].keys())
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in all_results:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

    print(f"\n✓ Resultados guardados en:")
    print(f"    {json_path}")
    print(f"    {csv_path}")

    return all_results


if __name__ == "__main__":
    run()
