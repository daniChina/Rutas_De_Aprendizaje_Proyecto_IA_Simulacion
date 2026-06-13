"""
run_fase2.py
============
Script de demostración de la Fase 2: evaluación semántica con el LLM.

Ejecutar desde la raíz del proyecto:
    python src/run_fase2.py

Requisitos previos:
    1. Tener el dataset generado (data/instances/instancia_A_pequena.json)
    2. Configurar OPENAI_API_KEY en el archivo .env

El script:
  1. Carga la instancia pequeña (Instancia A, 10 cursos).
  2. Evalúa cada curso con el LLM según un objetivo de ejemplo.
  3. Muestra el ranking de utilidades y los detalles de cada evaluación.
  4. Guarda el dataset enriquecido en data/processed/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.instance import load_instance
from src.llm import evaluar_problema, guardar_problema_evaluado, resumen_evaluacion
from src.llm.client import LLMClient

# Configurar logging con formato legible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_fase2")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Cargar instancia de prueba
    # ------------------------------------------------------------------
    instance_path = Path("data/instances/instancia_A_pequena.json")

    if not instance_path.exists():
        logger.error("Instancia no encontrada: %s", instance_path)
        logger.error("Ejecuta primero 'python src/run_example.py' para verificar Fase 1.")
        sys.exit(1)

    problema = load_instance(instance_path)
    logger.info("Instancia cargada: %s (%d cursos, T_max=%.0f h)",
                problema.instance_id, len(problema.courses), problema.t_max)

    # ------------------------------------------------------------------
    # 2. Definir el objetivo del usuario
    # ------------------------------------------------------------------
    objetivo = (
        "Quiero construir sistemas de inteligencia artificial aplicados al "
        "procesamiento de lenguaje natural, incluyendo el entrenamiento y "
        "despliegue de modelos de lenguaje grande (LLMs) en entornos de producción."
    )

    # ------------------------------------------------------------------
    # 3. Inicializar el cliente LLM
    #    EnvironmentError si OPENAI_API_KEY no está configurada
    # ------------------------------------------------------------------
    try:
        client = LLMClient(cache_path=".llm_cache.json")
    except EnvironmentError as e:
        logger.error("Credenciales no configuradas: %s", e)
        logger.error("Crea un archivo .env basándote en .env.example")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Evaluar el catálogo con el LLM
    # ------------------------------------------------------------------
    problema_evaluado = evaluar_problema(
        problema=problema,
        objetivo_usuario=objetivo,
        llm_client=client,
        delay_entre_llamadas=0.5,
        puntuacion_fallback=5,
    )

    # ------------------------------------------------------------------
    # 5. Mostrar resumen de evaluación
    # ------------------------------------------------------------------
    print(resumen_evaluacion(problema_evaluado, top_n=5))

    # ------------------------------------------------------------------
    # 6. Guardar dataset enriquecido para la Fase 3
    # ------------------------------------------------------------------
    ruta_salida = guardar_problema_evaluado(
        problema_evaluado,
        directorio="data/processed",
    )
    print(f"\nDataset enriquecido guardado en: {ruta_salida}")
    print("Proximo paso → Fase 3: ejecutar el algoritmo de optimizacion.")


if __name__ == "__main__":
    main()
