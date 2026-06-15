"""
pipeline.py
===========
Pipeline completo: entrada del usuario → secuencia óptima de cursos.

Toma como entrada:
  - Un archivo de instancia (JSON con cursos y T_max).
  - El objetivo de aprendizaje en lenguaje natural.

Produce como salida:
  - Una secuencia ordenada de cursos válida (respeta prerrequisitos).
  - Que maximiza la utilidad semántica total dentro de T_max horas.
  - Guardada en JSON y mostrada en consola con toda la información relevante.

Flujo interno:
  ┌─────────────────────────────────────────────────────┐
  │  ENTRADA                                            │
  │  instancia.json  +  objetivo (texto libre)          │
  └────────────────────┬────────────────────────────────┘
                       │
              [FASE 1] Carga y validación
                       │ load_instance()
                       │ validate_dag()
                       ▼
              LearningPathProblem (cursos sin u(v))
                       │
              [FASE 2] Evaluación semántica LLM
                       │ evaluar_problema()   ← con caché
                       │ u(v) ∈ [1,10] por curso
                       ▼
              LearningPathProblem (cursos con u(v))
                       │
              [FASE 3] Optimización híbrida
                       │ dp_knapsack_dag()    ← exacto si tratable
                       │ greedy_by_utility_density()
                       │ mc_path_sampler()
                       │ → mejor de los tres
                       ▼
              Secuencia óptima S* ⊆ V  (orden topológico)
                       │
              [FASE 4] Validación + enriquecimiento
                       │ is_valid_selection()
                       │ robustness_analysis()
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  SALIDA                                             │
  │  data/output/<instance_id>_ruta_optima.json         │
  │  + resumen en consola                               │
  └─────────────────────────────────────────────────────┘

Uso:
    # Modo interactivo (pide instancia y objetivo por stdin)
    python pipeline.py

    # Modo directo con argumentos
    python pipeline.py \\
        --instancia data/instances/instancia_B_mediana.json \\
        --objetivo  "Quiero especializarme en NLP y LLMs para trabajar en IA." \\
        --t_max     200 \\
        --mc_iter   2000

    # Forzar solo Monte Carlo (útil para instancias muy grandes)
    python pipeline.py --instancia ... --objetivo ... --force_mc

    # Usar caché LLM ya existente (no llama a la API si ya evaluó los cursos)
    python pipeline.py --instancia ... --objetivo ... --cache .llm_cache.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.llm.batch_evaluator import evaluar_problema_batch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
# Subir hasta encontrar src/ si el script está en un subdirectorio
for _candidate in [ROOT, ROOT.parent, ROOT.parent.parent]:
    if (_candidate / "src").exists():
        ROOT = _candidate
        break
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.llm.client import LLMClient
from src.llm.evaluator import evaluar_problema, guardar_problema_evaluado
from src.solver.llm_assisted import llm_assisted_solver
from src.solver.robustness import robustness_analysis

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

OUTPUT_DIR = ROOT / "data" / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de presentación
# ─────────────────────────────────────────────────────────────────────────────

def _banner(texto: str, ancho: int = 64, char: str = "═") -> None:
    pad = max(0, (ancho - len(texto) - 2) // 2)
    print(f"\n{char * ancho}")
    print(f"{char * pad} {texto} {char * pad}")
    print(char * ancho)


def _seccion(texto: str, ancho: int = 64) -> None:
    print(f"\n{'─' * ancho}")
    print(f"  {texto}")
    print(f"{'─' * ancho}")


def _imprimir_secuencia(cursos, problem, solver_usado: str) -> None:
    """Imprime la secuencia final de cursos en forma de tabla legible."""
    _seccion(f"SECUENCIA ÓPTIMA  [{solver_usado.upper()}]  — {len(cursos)} cursos")

    total_dur = 0.0
    total_util = 0.0

    print(f"  {'#':>2}  {'ID':<10}  {'Duración':>9}  {'Utilidad':>9}  Título")
    print(f"  {'─'*2}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*35}")

    for i, c in enumerate(cursos, start=1):
        prereqs_str = f"  ← {', '.join(c.prerrequisitos)}" if c.prerrequisitos else ""
        print(
            f"  {i:>2}.  {c.id:<10}  {c.duracion_horas:>7} h  "
            f"  {c.utility:>7.1f}    {c.titulo}{prereqs_str}"
        )
        total_dur  += c.duration
        total_util += c.utility

    print(f"  {'─'*2}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*35}")
    print(
        f"  {'TOTAL':>14}  {total_dur:>7.0f} h  "
        f"  {total_util:>7.1f}    "
        f"({total_dur / problem.t_max * 100:.1f}% del presupuesto)"
    )


def _imprimir_comparativa(result) -> None:
    """Muestra la tabla comparativa de los tres solvers."""
    _seccion("COMPARATIVA DE SOLVERS")

    print(f"  {'Solver':<14}  {'Utilidad':>9}  {'Cursos':>7}  {'Duración':>9}  {'u/h':>6}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*7}  {'─'*9}  {'─'*6}")

    def _fila(nombre, u, n, d):
        uph = u / d if d > 0 else 0
        marker = " ◄ MEJOR" if nombre.lower() in result.solver_used.lower() else ""
        print(f"  {nombre:<14}  {u:>9.1f}  {n:>7}  {d:>7.0f} h  {uph:>6.3f}{marker}")

    if result.dp_result:
        _fila("DP exacto",
              result.dp_result.objective_value,
              len(result.dp_result.selected_ids),
              result.dp_result.total_duration)
    else:
        print("  DP exacto       : omitido (problema demasiado grande para DP)")

    _fila("Greedy",
          result.greedy_result.objective_value,
          len(result.greedy_result.selected_ids),
          result.greedy_result.total_duration)

    _fila("Monte Carlo",
          result.mc_result.objective_value,
          len(result.mc_result.selected_ids),
          result.mc_result.total_duration)


def _imprimir_robustez(rob) -> None:
    """Muestra el análisis de robustez de la ruta seleccionada."""
    _seccion("ANÁLISIS DE ROBUSTEZ  (variabilidad ±20% en duraciones)")
    print(f"  P(completar dentro de T_max)  : {rob.p_feasible:.1%}")
    print(f"  Intervalo de confianza 95%    : [{rob.ci_95_lower:.1%}, {rob.ci_95_upper:.1%}]")
    print(f"  Duración media simulada       : {rob.mean_duration:.1f} ± {rob.std_duration:.1f} h")
    print(f"  Peor caso (percentil 95)      : {rob.percentile_95_dur:.1f} h")
    print(f"  Nivel de riesgo               : {rob.risk_level}")


# ─────────────────────────────────────────────────────────────────────────────
# Serialización de la salida
# ─────────────────────────────────────────────────────────────────────────────

def _guardar_salida(
    problem,
    cursos_seleccionados,
    result,
    rob,
    objetivo: str,
    t_pipeline: float,
    output_path: Path,
) -> None:
    """Guarda el resultado completo del pipeline en un JSON estructurado."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datos = {
        "metadata": {
            "timestamp":        datetime.now().isoformat(),
            "instancia":        problem.instance_id,
            "t_max_h":          problem.t_max,
            "objetivo_usuario": objetivo,
            "solver_usado":     result.solver_used,
            "tiempo_pipeline_s": round(t_pipeline, 3),
        },
        "solucion": {
            "n_cursos":           len(cursos_seleccionados),
            "utilidad_total":     result.objective_value,
            "duracion_total_h":   result.total_duration,
            "uso_presupuesto_pct": round(result.total_duration / problem.t_max * 100, 1),
            "utilidad_por_hora":  round(result.utilidad_por_hora, 4),
            "es_valida":          problem.is_valid_selection(result.selected_ids),
            "secuencia": [
                {
                    "orden":             i,
                    "id":                c.id,
                    "titulo":            c.titulo,
                    "duracion_horas":    c.duracion_horas,
                    "utilidad":          c.utility,
                    "prerrequisitos":    c.prerrequisitos,
                    "justificacion_llm": c.justificacion or "",
                }
                for i, c in enumerate(cursos_seleccionados, start=1)
            ],
        },
        "robustez": {
            "cv":              0.20,
            "n_simulaciones":  5000,
            "p_feasible":      rob.p_feasible,
            "ic_95_lower":     rob.ci_95_lower,
            "ic_95_upper":     rob.ci_95_upper,
            "mean_dur_h":      rob.mean_duration,
            "std_dur_h":       rob.std_duration,
            "p95_dur_h":       rob.percentile_95_dur,
            "nivel_riesgo":    rob.risk_level,
        },
        "comparativa_solvers": {
            "dp": {
                "utilidad":  result.dp_result.objective_value if result.dp_result else None,
                "duracion_h": result.dp_result.total_duration if result.dp_result else None,
                "n_cursos":  len(result.dp_result.selected_ids) if result.dp_result else None,
            },
            "greedy": {
                "utilidad":  result.greedy_result.objective_value,
                "duracion_h": result.greedy_result.total_duration,
                "n_cursos":  len(result.greedy_result.selected_ids),
            },
            "monte_carlo": {
                "utilidad":  result.mc_result.objective_value,
                "duracion_h": result.mc_result.total_duration,
                "n_cursos":  len(result.mc_result.selected_ids),
                "n_iter":    result.mc_result.n_iterations,
            },
        },
        "catalogo_evaluado": [
            {
                "id":             c.id,
                "titulo":         c.titulo,
                "duracion_horas": c.duracion_horas,
                "utilidad":       c.utility,
                "en_solucion":    c.id in result.selected_ids,
                "justificacion":  c.justificacion or "",
            }
            for c in sorted(problem.courses,
                            key=lambda x: x.utility or 0, reverse=True)
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    logger.info("Resultado guardado en: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    instancia_path: str | Path,
    objetivo_usuario: str,
    t_max_override: Optional[float] = None,
    cache_path: str = ".llm_cache.json",
    mc_iter: int = 2000,
    mc_temperature: float = 1.0,
    mc_seed: int = 42,
    force_mc: bool = False,
    n_simulaciones_robustez: int = 5000,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict:
    """
    Ejecuta el pipeline completo y devuelve la secuencia óptima de cursos.

    Args:
        instancia_path:          Ruta al JSON de la instancia (cursos + T_max).
        objetivo_usuario:        Meta de aprendizaje en lenguaje natural.
        t_max_override:          Si se indica, sobreescribe el T_max de la instancia.
        cache_path:              Ruta al archivo de caché LLM (evita llamadas repetidas).
        mc_iter:                 Iteraciones del solver Monte Carlo.
        mc_temperature:          Temperatura del muestreo MC (1.0 = recomendado).
        mc_seed:                 Semilla para reproducibilidad.
        force_mc:                Si True, usa solo MC (omite DP exacto).
        n_simulaciones_robustez: Simulaciones Gamma para el análisis de robustez.
        output_dir:              Directorio donde se guarda el JSON de salida.

    Returns:
        Diccionario con la secuencia, métricas y ruta del archivo guardado.
    """
    t_inicio = time.perf_counter()

    _banner("PIPELINE DE RUTAS DE APRENDIZAJE ÓPTIMAS")
    print(f"  Instancia  : {instancia_path}")
    print(f"  Objetivo   : {objetivo_usuario[:80]}{'...' if len(objetivo_usuario) > 80 else ''}")
    print(f"  Caché LLM  : {cache_path}")
    print(f"  MC iter    : {mc_iter} | temp={mc_temperature} | seed={mc_seed}")

    # ── FASE 1: Carga y validación ────────────────────────────────────────────
    _seccion("FASE 1 — Carga y validación del grafo")

    problema = load_instance(instancia_path)

    if t_max_override is not None:
        problema.t_max = float(t_max_override)
        logger.info("T_max sobreescrito a %.0f h", problema.t_max)

    print(problema.summary())

    is_dag, ciclos = problema.validate_dag()
    if not is_dag:
        raise ValueError(
            f"El grafo de prerrequisitos contiene ciclos en: {ciclos}. "
            "Revisa el archivo de instancia antes de continuar."
        )
    print("\n  ✓ DAG válido — no hay ciclos en los prerrequisitos.")
    # ── FASE 2: Evaluación semántica con LLM ─────────────────────────────────
    
    _seccion("FASE 2 — Evaluación semántica con LLM")
    ya_evaluados = sum(1 for c in problema.courses if c.utilidad_relativa is not None)
    pendientes   = len(problema.courses) - ya_evaluados

    if pendientes == 0:
        print(f"  ✓ Todos los {len(problema.courses)} cursos ya tienen utilidad. "
          "Usando caché — sin llamadas a la API.")
    else:
      print(f"  Cursos ya evaluados : {ya_evaluados}")
      print(f"  Cursos pendientes   : {pendientes} → llamando al LLM")

    try:
     client = LLMClient(cache_path=cache_path)
    except EnvironmentError as e:
      logger.error("Credenciales LLM no configuradas: %s", e)
      logger.error("Configura GEMINI_API_KEY o OPENAI_API_KEY en el archivo .env")
      raise
  
    t_fase2 = time.perf_counter()

    # Decidir qué evaluador usar según el tamaño de la instancia (número de cursos)
    UMBRAL_BATCH = 20   # usar batch evaluator si hay más de 20 cursos pendientes

    if len(problema.courses) > UMBRAL_BATCH and pendientes > 0:
      logger.info("Instancia grande (%d cursos) → usando evaluador por lotes (batch)", len(problema.courses))
      problema = evaluar_problema_batch(
        problema=problema,
        objetivo_usuario=objetivo_usuario,
        llm_client=client,
        batch_size=35,                # puedes ajustar, 35 es el máximo para instancia C
        puntuacion_fallback=5,
        fallback_a_individual=True,   # si un lote falla, re-intenta curso a curso
    )
    else:
     logger.info("Instancia pequeña (%d cursos) → usando evaluador individual", len(problema.courses))
     problema = evaluar_problema(
        problema=problema,
        objetivo_usuario=objetivo_usuario,
        llm_client=client,
        delay_entre_llamadas=0.3,
        puntuacion_fallback=5,
    )

    t_fase2 = time.perf_counter() - t_fase2

    evaluados_ok = sum(1 for c in problema.courses
                       if c.utilidad_relativa is not None
                       and not (c.justificacion or "").startswith("[FALLBACK]"))
    print(f"\n  ✓ Evaluación completa en {t_fase2:.1f}s "
          f"({evaluados_ok}/{len(problema.courses)} exitosos)")

    # Guardar dataset enriquecido como paso intermedio
    guardar_problema_evaluado(problema, directorio=ROOT / "data" / "processed")

    # ── FASE 3: Optimización híbrida ─────────────────────────────────────────
    _seccion("FASE 3 — Optimización híbrida (DP + Greedy + Monte Carlo)")

    t_fase3 = time.perf_counter()
    result = llm_assisted_solver(
        problem=problema,
        objetivo_usuario=objetivo_usuario,
        llm_client=client,
        mc_iterations=mc_iter,
        mc_temperature=mc_temperature,
        mc_seed=mc_seed,
        force_mc_only=force_mc,
    )
    t_fase3 = time.perf_counter() - t_fase3

    print(f"\n  ✓ Optimización completada en {t_fase3:.1f}s")
    print(f"  Mejor solver    : {result.solver_used}")
    print(f"  Utilidad total  : {result.objective_value:.1f}")
    print(f"  Duración total  : {result.total_duration:.0f} h / {problema.t_max:.0f} h "
          f"({result.total_duration / problema.t_max * 100:.1f}%)")

    # ── FASE 4: Validación y enriquecimiento ──────────────────────────────────
    _seccion("FASE 4 — Validación y análisis de robustez")

    # Validar que la solución es factible
    es_valida = problema.is_valid_selection(result.selected_ids)
    if not es_valida:
        logger.error("¡La solución encontrada NO es válida! Revisar el solver.")
        raise RuntimeError("La solución del optimizador viola restricciones de tiempo o prerrequisitos.")

    print(f"  ✓ Solución válida: cumple prerrequisitos y restricción de tiempo.")

    # Obtener cursos en orden topológico (secuencia ejecutable)
    topo_order = problema.topological_order()
    cursos_secuencia = [
        problema.get_course(cid)
        for cid in topo_order
        if cid in result.selected_ids
    ]

    # Análisis de robustez
    rob = robustness_analysis(
        problem=problema,
        selected_ids=result.selected_ids,
        n_simulations=n_simulaciones_robustez,
        cv=0.20,
        seed=mc_seed,
    )

    # ── Mostrar resultados ────────────────────────────────────────────────────
    _imprimir_comparativa(result)
    _imprimir_secuencia(cursos_secuencia, problema, result.solver_used)
    _imprimir_robustez(rob)

    # ── Guardar salida ────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_inicio
    output_path = (
        Path(output_dir) /
        f"{problema.instance_id}_ruta_optima.json"
    )
    _guardar_salida(
        problema, cursos_secuencia, result, rob,
        objetivo_usuario, t_total, output_path,
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    _banner("RESULTADO FINAL")
    print(f"  Secuencia de {len(cursos_secuencia)} cursos en orden de estudio:")
    print()
    for i, c in enumerate(cursos_secuencia, start=1):
        print(f"    {i:>2}. [{c.id}] {c.titulo}  ({c.duracion_horas} h | u={c.utility:.0f}/10)")
    print()
    print(f"  Utilidad total     : {result.objective_value:.1f} / "
          f"{sum(c.utility for c in problema.courses):.1f} máx. posible")
    print(f"  Horas totales      : {result.total_duration:.0f} h / {problema.t_max:.0f} h")
    print(f"  P(completar ruta)  : {rob.p_feasible:.1%}  → {rob.risk_level}")
    print(f"  Tiempo del pipeline: {t_total:.1f} s")
    print(f"  Resultado guardado : {output_path}")
    print()

    return {
        "secuencia":         [c.id for c in cursos_secuencia],
        "cursos":            cursos_secuencia,
        "utilidad_total":    result.objective_value,
        "duracion_total_h":  result.total_duration,
        "solver_usado":      result.solver_used,
        "p_feasible":        rob.p_feasible,
        "es_valida":         es_valida,
        "output_path":       str(output_path),
        "tiempo_pipeline_s": round(t_total, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline completo: instancia JSON + objetivo → secuencia óptima de cursos.\n"
            "Ejemplo:\n"
            "  python pipeline.py \\\n"
            "    --instancia data/instances/instancia_B_mediana.json \\\n"
            "    --objetivo  \"Quiero dominar NLP y LLMs para trabajar en IA.\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instancia", "-i",
        help="Ruta al archivo JSON de la instancia.",
    )
    parser.add_argument(
        "--objetivo", "-o",
        help="Objetivo de aprendizaje en lenguaje natural.",
    )
    parser.add_argument(
        "--t_max", type=float, default=None,
        help="Sobreescribir T_max de la instancia (horas).",
    )
    parser.add_argument(
        "--cache", default=".llm_cache.json",
        help="Ruta al archivo de caché LLM (default: .llm_cache.json).",
    )
    parser.add_argument(
        "--mc_iter", type=int, default=2000,
        help="Iteraciones Monte Carlo (default: 2000).",
    )
    parser.add_argument(
        "--mc_temp", type=float, default=1.0,
        help="Temperatura Monte Carlo (default: 1.0).",
    )
    parser.add_argument(
        "--mc_seed", type=int, default=42,
        help="Semilla aleatoria (default: 42).",
    )
    parser.add_argument(
        "--force_mc", action="store_true",
        help="Forzar solo Monte Carlo (omitir DP exacto).",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help=f"Directorio de salida (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def _pedir_interactivo() -> tuple[str, str]:
    """Pide instancia y objetivo al usuario por stdin."""
    print("\nNo se proporcionaron argumentos. Modo interactivo.\n")

    instances_dir = ROOT / "data" / "instances"
    instancias_disponibles = sorted(instances_dir.glob("*.json")) if instances_dir.exists() else []

    if instancias_disponibles:
        print("Instancias disponibles:")
        for i, p in enumerate(instancias_disponibles, start=1):
            print(f"  [{i}] {p}")
        print()
        entrada = input("Selecciona número o escribe la ruta completa: ").strip()
        try:
            idx = int(entrada) - 1
            instancia = str(instancias_disponibles[idx])
        except (ValueError, IndexError):
            instancia = entrada
    else:
        instancia = input("Ruta a la instancia JSON: ").strip()

    print()
    objetivo = input(
        "Objetivo de aprendizaje (describe en lenguaje natural qué quieres lograr):\n> "
    ).strip()

    if not objetivo:
        objetivo = (
            "Quiero especializarme en inteligencia artificial con foco en "
            "procesamiento de lenguaje natural y modelos de lenguaje grande."
        )
        print(f"[Usando objetivo por defecto: {objetivo}]")

    return instancia, objetivo


if __name__ == "__main__":
    args = _parse_args()

    # Modo interactivo si no se pasan argumentos obligatorios
    if not args.instancia or not args.objetivo:
        instancia, objetivo = _pedir_interactivo()
    else:
        instancia = args.instancia
        objetivo  = args.objetivo

    try:
        run_pipeline(
            instancia_path=instancia,
            objetivo_usuario=objetivo,
            t_max_override=args.t_max,
            cache_path=args.cache,
            mc_iter=args.mc_iter,
            mc_temperature=args.mc_temp,
            mc_seed=args.mc_seed,
            force_mc=args.force_mc,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError:
        print(
            "\n[ERROR] Credenciales LLM no encontradas.\n"
            "Crea un archivo .env con GEMINI_API_KEY o OPENAI_API_KEY.\n"
            "Consulta .env.example para ver el formato.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[Cancelado por el usuario]")
        sys.exit(0)
