"""
experimento_2_temperatura.py
============================
Experimento 2 — Impacto de la temperatura en diversidad y utilidad

Hipótesis:
    - Temperatura baja → el sampler se comporta como greedy (explota el mejor
      curso disponible), produce soluciones similares entre sí (baja diversidad)
      con utilidad alta pero propensa a quedar atrapada en óptimos locales.
    - Temperatura alta → muestreo casi uniforme, mayor diversidad de soluciones
      pero utilidad promedio más baja.
    - Temperatura ≈ 1.0 ofrece el mejor balance exploración/explotación.

Métricas de diversidad:
    1. Jaccard medio: promedio de similitudes Jaccard entre pares de soluciones
       distintas producidas con la misma temperatura. Valor 1 = todas iguales.
    2. Varianza de utilidad: dispersión de la utilidad entre réplicas.
    3. Cardinalidad media: número medio de cursos en la solución.

Diseño:
    - Instancia fija: instancia_B_mediana.
    - n_iter fijo en 1000 (suficiente para converger según Exp. 1).
    - Temperatura ∈ {0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0}.
    - N_REPS réplicas por temperatura para estimar diversidad.

Salida:
    results/experimento_2_temperatura.json
    results/experimento_2_temperatura.csv
"""

from __future__ import annotations

import json
import logging
import sys
import time
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev, variance
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.solver.mc_sampler import mc_path_sampler

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("exp2")

# ── Parámetros ────────────────────────────────────────────────────────────────
INSTANCE_PATH = ROOT / "data/processed/instancia_B_evaluada.json"
TEMPERATURES    = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
N_ITER          = 1000
N_REPS          = 15   # más réplicas para estimar bien la diversidad
BASE_SEED       = 42
RESULTS_DIR     = ROOT / "results"


def jaccard(set_a: set, set_b: set) -> float:
    """Similitud de Jaccard entre dos conjuntos de IDs de cursos."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def jaccard_medio(solutions: list[list[str]]) -> float:
    """Promedio de Jaccard sobre todos los pares de soluciones."""
    sets = [set(s) for s in solutions]
    pares = list(combinations(range(len(sets)), 2))
    if not pares:
        return 1.0
    scores = [jaccard(sets[i], sets[j]) for i, j in pares]
    return mean(scores)


def run() -> list[dict[str, Any]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    problem = load_instance(INSTANCE_PATH)

    if any(c.utilidad_relativa is None for c in problem.courses):
        raise RuntimeError("Instancia sin utilidades. Ejecuta primero la Fase 2.")

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO 2 — Impacto de la temperatura en diversidad y utilidad")
    print(f"Instancia : {problem.instance_id} | n_iter={N_ITER} | réplicas={N_REPS}")
    print(f"{'='*70}")
    print(
        f"{'temp':>5}  {'u_media':>8}  {'u_std':>7}  {'jaccard':>8}  "
        f"{'u_min':>7}  {'u_max':>7}  {'cards_media':>12}  {'t_medio_s':>10}"
    )
    print(f"{'─'*5}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*12}  {'─'*10}")

    all_results: list[dict[str, Any]] = []

    for temp in TEMPERATURES:
        utilidades:   list[float] = []
        tiempos:      list[float] = []
        soluciones:   list[list[str]] = []
        cardinalidad: list[int] = []

        for rep in range(N_REPS):
            seed = BASE_SEED + rep * 73

            t0 = time.perf_counter()
            result = mc_path_sampler(
                problem,
                n_iterations=N_ITER,
                temperature=temp,
                seed=seed,
            )
            elapsed = time.perf_counter() - t0

            utilidades.append(result.objective_value)
            tiempos.append(elapsed)
            soluciones.append(result.selected_ids)
            cardinalidad.append(len(result.selected_ids))

        u_media = mean(utilidades)
        u_std   = stdev(utilidades) if len(utilidades) > 1 else 0.0
        j_medio = jaccard_medio(soluciones)
        card_m  = mean(cardinalidad)
        t_medio = mean(tiempos)

        print(
            f"{temp:>5.1f}  {u_media:>8.2f}  {u_std:>7.3f}  {j_medio:>8.4f}  "
            f"{min(utilidades):>7.2f}  {max(utilidades):>7.2f}  "
            f"{card_m:>12.1f}  {t_medio:>10.4f}"
        )

        all_results.append({
            "temperatura":          temp,
            "n_iter":               N_ITER,
            "n_reps":               N_REPS,
            "utilidad_media":       round(u_media, 4),
            "utilidad_std":         round(u_std, 4),
            "utilidad_min":         round(min(utilidades), 4),
            "utilidad_max":         round(max(utilidades), 4),
            "jaccard_medio":        round(j_medio, 6),
            "diversidad":           round(1 - j_medio, 6),  # 0=todo igual, 1=máximo distinto
            "cardinalidad_media":   round(card_m, 2),
            "cardinalidad_std":     round(stdev(cardinalidad) if len(cardinalidad) > 1 else 0.0, 3),
            "tiempo_medio_s":       round(t_medio, 6),
            "instancia":            problem.instance_id,
            "n_cursos":             len(problem.courses),
            "t_max":                problem.t_max,
        })

    # ── Persistencia ──────────────────────────────────────────────────────────
    json_path = RESULTS_DIR / "experimento_2_temperatura.json"
    csv_path  = RESULTS_DIR / "experimento_2_temperatura.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experimento":  2,
                "descripcion":  "Impacto de la temperatura en diversidad y utilidad",
                "resultados":   all_results,
            },
            f, ensure_ascii=False, indent=2,
        )

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
