"""
run_fase2_batch.py
==================
Versión eficiente de run_fase2.py para instancias grandes (Instancia C).

Evalúa todos los cursos en UNA sola llamada al LLM en lugar de una por curso,
reduciendo el consumo de tokens de ~64.000 a ~6.000 (~90% de ahorro).

Uso:
    # Instancia C (recomendado para ahorrar tokens)
    python src/run_fase2_batch.py

    # Cualquier instancia con argumento
    python src/run_fase2_batch.py --instancia data/instances/instancia_C_grande.json

    # Comparar consumo estimado antes de ejecutar
    python src/run_fase2_batch.py --estimar

Cuándo usar este script vs run_fase2.py:
    ┌─────────────────────────────────┬───────────────┬────────────────────┐
    │ Situación                       │ run_fase2.py  │ run_fase2_batch.py │
    ├─────────────────────────────────┼───────────────┼────────────────────┤
    │ Instancia A o B (≤ 20 cursos)   │ ✓             │ también sirve      │
    │ Instancia C (35 cursos)         │ agota tokens  │ ✓ recomendado      │
    │ Re-evaluar solo cursos nuevos   │ ✓ (usa caché) │ re-evalúa todos    │
    │ Cuota de tokens muy ajustada    │               │ ✓                  │
    └─────────────────────────────────┴───────────────┴────────────────────┘
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _c in [ROOT, ROOT.parent, ROOT.parent.parent]:
    if (_c / "src").exists():
        ROOT = _c
        break
sys.path.insert(0, str(ROOT))

from src.instance import load_instance
from src.llm.client import LLMClient
from src.llm.evaluator import guardar_problema_evaluado, resumen_evaluacion
from src.llm.batch_evaluator import evaluar_problema_batch, BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_fase2_batch")

DEFAULT_INSTANCE = ROOT / "data/instances/instancia_C_grande.json"
OBJETIVO_DEFAULT = (
    "Quiero construir sistemas de inteligencia artificial aplicados al "
    "procesamiento de lenguaje natural, incluyendo el entrenamiento y "
    "despliegue de modelos de lenguaje grande (LLMs) en entornos de producción."
)


def estimar_tokens(n_cursos: int) -> None:
    """Imprime una comparativa del consumo estimado de tokens."""
    # Estimaciones basadas en mediciones reales del sistema
    tokens_individual = n_cursos * (1302 + 172 + 350)   # system + user + output por curso
    tokens_batch      = 1302 + (n_cursos * 55) + (n_cursos * 100)   # system + input batch + output batch
    ahorro            = tokens_individual - tokens_batch
    pct               = ahorro / tokens_individual * 100

    print(f"\n{'─'*55}")
    print(f"  Estimación de consumo de tokens para {n_cursos} cursos")
    print(f"{'─'*55}")
    print(f"  Modo individual (run_fase2.py)  : ~{tokens_individual:>7,} tokens")
    print(f"  Modo batch     (este script)    : ~{tokens_batch:>7,} tokens")
    print(f"  Ahorro                          : ~{ahorro:>7,} tokens ({pct:.0f}%)")
    print(f"{'─'*55}\n")


def main(instancia_path: Path, objetivo: str, solo_estimar: bool = False) -> None:
    if not instancia_path.exists():
        logger.error("Instancia no encontrada: %s", instancia_path)
        logger.error("Genera primero las instancias con: python experiments/generate_instances.py")
        sys.exit(1)

    problema = load_instance(instancia_path)
    n = len(problema.courses)

    logger.info("Instancia cargada: %s (%d cursos, T_max=%.0f h)",
                problema.instance_id, n, problema.t_max)

    estimar_tokens(n)

    if solo_estimar:
        print("  (modo --estimar: sin llamadas reales al LLM)")
        return

    # Verificar cuántos cursos ya tienen utilidad (de ejecuciones previas)
    ya_evaluados = sum(1 for c in problema.courses if c.utilidad_relativa is not None)
    if ya_evaluados > 0:
        logger.info("%d/%d cursos ya evaluados. Se evaluarán los %d restantes.",
                    ya_evaluados, n, n - ya_evaluados)

    # Inicializar cliente LLM (sin caché para batch — los prompts son distintos)
    try:
        client = LLMClient(cache_path=None)
    except EnvironmentError as e:
        logger.error("Credenciales no configuradas: %s", e)
        logger.error("Crea un archivo .env con GEMINI_API_KEY o OPENAI_API_KEY")
        sys.exit(1)

    logger.info("Proveedor activo: %s | modelo: %s", client.provider, client.model)

    # ── Evaluación batch ──────────────────────────────────────────────────────
    problema_evaluado = evaluar_problema_batch(
        problema=problema,
        objetivo_usuario=objetivo,
        llm_client=client,
        batch_size=BATCH_SIZE,
        puntuacion_fallback=5,
        fallback_a_individual=True,   # re-intenta individualmente si algo falla
    )

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(resumen_evaluacion(problema_evaluado, top_n=5))

    # ── Guardar ───────────────────────────────────────────────────────────────
    ruta_salida = guardar_problema_evaluado(
        problema_evaluado,
        directorio=ROOT / "data/processed",
    )
    print(f"\nDataset enriquecido guardado en: {ruta_salida}")
    print("Próximo paso → Fase 3: python pipeline.py --instancia "
          f"{instancia_path} --objetivo '...'")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación batch del LLM — consume ~90% menos tokens que el modo individual."
    )
    parser.add_argument(
        "--instancia", "-i",
        type=Path,
        default=DEFAULT_INSTANCE,
        help=f"Ruta a la instancia JSON (default: {DEFAULT_INSTANCE.name}).",
    )
    parser.add_argument(
        "--objetivo", "-o",
        default=OBJETIVO_DEFAULT,
        help="Objetivo de aprendizaje (default: NLP/LLMs).",
    )
    parser.add_argument(
        "--estimar",
        action="store_true",
        help="Solo estima el consumo de tokens sin llamar al LLM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.instancia, args.objetivo, solo_estimar=args.estimar)
