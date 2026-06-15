"""
experimento_4_robustez.py
=========================
Experimento 4 — Robustez: P(factibilidad) para distintas rutas

Hipótesis:
    - La ruta óptima del DP puede ser frágil: al introducir variabilidad en
      las duraciones, su probabilidad de factibilidad puede caer significativa-
      mente si usa casi todo el presupuesto disponible.
    - Rutas más cortas (menor utilidad pero más holgura) tienen mayor
      P(factibilidad) bajo incertidumbre.
    - Existe un trade-off entre utilidad y robustez que es relevante para
      recomendar rutas al usuario.

Diseño:
    Para cada instancia disponible se generan K rutas candidatas con distintas
    estrategias de selección:
        1. Ruta DP óptima (máxima utilidad, mínima holgura).
        2. Ruta Greedy densidad.
        3. Ruta MC mejor.
        4. Rutas parciales (50%, 70%, 90% del presupuesto DP) — generadas
           truncando la ruta DP óptima por orden topológico.

    Para cada ruta se ejecuta robustness_analysis() con:
        - cv ∈ {0.10, 0.20, 0.30}  (variabilidad baja, media, alta)
        - M = 5000 simulaciones por ruta × cv

    Métricas registradas:
        utilidad, duración_nominal, P(factibilidad), IC_95%, percentil_95_dur,
        nivel_riesgo, cv.

Salida:
    results/experimento_4_robustez.json
    results/experimento_4_robustez.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.problem import LearningPathProblem
from src.solver.baseline import dp_knapsack_dag, greedy_by_utility_density
from src.solver.mc_sampler import mc_path_sampler
from src.solver.robustness import robustness_analysis, sensitivity_cv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("exp4")

# ── Parámetros ────────────────────────────────────────────────────────────────
INSTANCES = [
    ("A", ROOT / "data/processed/instancia_A_evaluada.json"),
    ("B", ROOT / "data/processed/instancia_B_evaluada.json"),
    ("C", ROOT / "data/processed/instancia_C_evaluada.json"),
]
CV_VALUES     = [0.10, 0.20, 0.30]
N_SIMULATIONS = 5000
N_ITER_MC     = 2000
SEED          = 42
MAX_DP_STATES = 500_000
RESULTS_DIR   = ROOT / "results"


def _generate_partial_routes(
    problem: LearningPathProblem,
    full_route: list[str],
    fractions: list[float] = [0.5, 0.7, 0.9],
) -> dict[str, list[str]]:
    """
    Genera rutas parciales tomando los primeros cursos del orden topológico
    de la ruta completa hasta alcanzar fraction × T_max.
    """
    topo = problem.topological_order()
    # Ordenar full_route según el orden topológico
    full_topo = [cid for cid in topo if cid in set(full_route)]

    partials = {}
    for frac in fractions:
        budget = problem.t_max * frac
        partial: list[str] = []
        acum = 0.0
        for cid in full_topo:
            c = problem.get_course(cid)
            if c and acum + c.duration <= budget:
                partial.append(cid)
                acum += c.duration
        label = f"parcial_{int(frac*100)}pct"
        partials[label] = partial

    return partials


def _analyze_route(
    problem: LearningPathProblem,
    route_ids: list[str],
    route_label: str,
    instance_label: str,
    cv_values: list[float],
) -> list[dict[str, Any]]:
    """Ejecuta robustness_analysis para una ruta con cada valor de cv."""
    if not route_ids:
        return []

    u = problem.objective_value(route_ids)
    d = problem.selection_duration(route_ids)
    holgura = problem.t_max - d

    rows: list[dict[str, Any]] = []

    for cv in cv_values:
        rob = robustness_analysis(
            problem,
            selected_ids=route_ids,
            n_simulations=N_SIMULATIONS,
            cv=cv,
            seed=SEED,
        )

        row: dict[str, Any] = {
            "instancia_label":    instance_label,
            "instancia_id":       problem.instance_id,
            "ruta":               route_label,
            "n_cursos_ruta":      len(route_ids),
            "utilidad":           round(u, 4),
            "duracion_nominal_h": round(d, 2),
            "holgura_h":          round(holgura, 2),
            "holgura_pct":        round(holgura / problem.t_max * 100, 2),
            "t_max":              problem.t_max,
            "cv":                 cv,
            "n_simulaciones":     N_SIMULATIONS,
            "p_factible":         round(rob.p_feasible, 6),
            "p_factible_pct":     round(rob.p_feasible * 100, 2),
            "ic95_lower":         round(rob.ci_95_lower, 6),
            "ic95_upper":         round(rob.ci_95_upper, 6),
            "dur_media_sim":      round(rob.mean_duration, 2),
            "dur_std_sim":        round(rob.std_duration, 2),
            "dur_p95_sim":        round(rob.percentile_95_dur, 2),
            "nivel_riesgo":       rob.risk_level.strip(),
        }
        rows.append(row)

        print(
            f"   [{route_label:20s}] cv={cv:.2f}  "
            f"u={u:.1f}  d={d:.0f}h  holgura={holgura:.0f}h  "
            f"P(fact)={rob.p_feasible:.1%}  [{rob.risk_level.strip()}]"
        )

    return rows


def run() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENTO 4 — Robustez: P(factibilidad) para distintas rutas")
    print(f"cv ∈ {CV_VALUES} | M={N_SIMULATIONS} simulaciones por ruta")
    print(f"{'='*80}")

    all_rows: list[dict[str, Any]] = []

    for label, path in INSTANCES:
        if not path.exists():
            print(f"\n⚠  Instancia {label} no encontrada. Saltando.")
            continue

        problem = load_instance(path)

        if any(c.utilidad_relativa is None for c in problem.courses):
            print(f"\n⚠  Instancia {label} sin utilidades. Saltando.")
            continue

        print(f"\n── Instancia {label}: {problem.instance_id} ──")
        print(f"   Nodos={len(problem.courses)} | T_max={problem.t_max:.0f}h")

        routes: dict[str, list[str]] = {}

        # ── 1. Ruta DP (si es tratable) ───────────────────────────────────────
        n, W = len(problem.courses), int(problem.t_max)
        if n * W <= MAX_DP_STATES:
            dp_res = dp_knapsack_dag(problem)
            routes["dp_optima"] = dp_res.selected_ids
            # Rutas parciales derivadas del DP
            parciales = _generate_partial_routes(problem, dp_res.selected_ids)
            routes.update(parciales)
        else:
            print(f"   DP omitido (n×W={n*W:,} > {MAX_DP_STATES:,})")

        # ── 2. Ruta Greedy ────────────────────────────────────────────────────
        gr_res = greedy_by_utility_density(problem)
        routes["greedy"] = gr_res.selected_ids

        # ── 3. Mejor ruta Monte Carlo ─────────────────────────────────────────
        mc_res = mc_path_sampler(problem, n_iterations=N_ITER_MC, seed=SEED)
        routes["mc_mejor"] = mc_res.selected_ids

        # ── Análisis de robustez para cada ruta ───────────────────────────────
        for route_label, route_ids in routes.items():
            rows = _analyze_route(
                problem, route_ids, route_label, label, CV_VALUES
            )
            all_rows.extend(rows)

    # ── Tabla de sensibilidad adicional (instancia B, ruta DP) ───────────────
    print(f"\n── Análisis de sensibilidad cv vs P(factible) — Instancia B ──")
    inst_b_path = ROOT / "data/processed/instancia_B_evaluada.json"
    if inst_b_path.exists():
        prob_b = load_instance(inst_b_path)
        if not any(c.utilidad_relativa is None for c in prob_b.courses):
            n_b, W_b = len(prob_b.courses), int(prob_b.t_max)
            if n_b * W_b <= MAX_DP_STATES:
                dp_b = dp_knapsack_dag(prob_b)
                sens = sensitivity_cv(
                    prob_b,
                    dp_b.selected_ids,
                    cv_values=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
                    n_simulations=3000,
                    seed=SEED,
                )
                print(f"   {'cv':>5}  {'P(factible)':>12}")
                for cv_v, p in sens.items():
                    print(f"   {cv_v:>5.2f}  {p:>11.1%}")

                # Guardar sensibilidad como JSON aparte
                sens_path = RESULTS_DIR / "experimento_4_sensibilidad_cv.json"
                with sens_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "experimento":   "4b",
                            "descripcion":   "Sensibilidad P(factible) vs cv — ruta DP instancia B",
                            "instancia":     prob_b.instance_id,
                            "ruta":          "dp_optima",
                            "n_simulaciones": 3000,
                            "resultados":    [
                                {"cv": cv_v, "p_factible": p} for cv_v, p in sens.items()
                            ],
                        },
                        f, ensure_ascii=False, indent=2,
                    )
                print(f"\n✓ Sensibilidad guardada en: {sens_path}")

    # ── Persistencia principal ────────────────────────────────────────────────
    json_path = RESULTS_DIR / "experimento_4_robustez.json"
    csv_path  = RESULTS_DIR / "experimento_4_robustez.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experimento":   4,
                "descripcion":   "Robustez P(factibilidad) para distintas rutas y cv",
                "resultados":    all_rows,
            },
            f, ensure_ascii=False, indent=2,
        )

    if all_rows:
        headers = list(all_rows[0].keys())
        with csv_path.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in all_rows:
                f.write(",".join(str(row[h]) for h in headers) + "\n")

    print(f"\n✓ Resultados guardados en:")
    print(f"    {json_path}")
    print(f"    {csv_path}")

    return all_rows


if __name__ == "__main__":
    run()
