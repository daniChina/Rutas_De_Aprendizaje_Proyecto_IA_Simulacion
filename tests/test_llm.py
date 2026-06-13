"""
test_llm.py  (src/test_llm.py)
================================
Script de verificación de conexión al proveedor LLM activo.

Ejecutar desde la raíz del proyecto:
    python src/test_llm.py

Verifica:
  1. Que las variables de entorno están correctamente configuradas.
  2. Que el proveedor responde con JSON válido.
  3. Que Pydantic valida la respuesta correctamente.

Si todo está bien, imprime el proveedor activo, el modelo usado
y la evaluación de prueba devuelta por el LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import LLMClient, LLM_PROVIDER


# Curso y objetivo de prueba (no consume el catálogo real)
_CURSO_PRUEBA = {
    "id": "TEST_001",
    "titulo": "Fundamentos de Programación en Python",
    "descripcion": (
        "Introduce los conceptos esenciales de la programación imperativa "
        "usando Python: variables, tipos de datos, estructuras de control, "
        "funciones y módulos. Se practican ejercicios de lógica computacional "
        "y se trabaja con entornos Jupyter Notebook."
    ),
}

_OBJETIVO_PRUEBA = (
    "Quiero aprender machine learning y redes neuronales desde cero "
    "para trabajar como científico de datos en la industria."
)


def main() -> None:
    print("=" * 55)
    print("Test de conexión al LLM")
    print("=" * 55)

    try:
        client = LLMClient()
        print(f"  Proveedor : {client.provider}")
        print(f"  Modelo    : {client.model}")
    except EnvironmentError as e:
        print(f"\n✗ Error de configuración:\n  {e}")
        print("\nPasos para solucionarlo:")
        print("  1. Copia .env.example a .env")
        print("  2. Configura LLM_PROVIDER y la clave correspondiente")
        sys.exit(1)

    print(f"\nEnviando curso de prueba: '{_CURSO_PRUEBA['titulo']}'")
    print(f"Objetivo: {_OBJETIVO_PRUEBA[:80]}…\n")

    evaluacion = client.evaluar_curso(_CURSO_PRUEBA, _OBJETIVO_PRUEBA)

    if evaluacion is None:
        print("✗ El LLM no devolvió una respuesta válida.")
        print("  Revisa los logs anteriores para más detalles.")
        sys.exit(1)

    print("✓ Conexión exitosa. Respuesta recibida y validada:\n")
    print(f"  curso_id           : {evaluacion.curso_id}")
    print(f"  utilidad_relativa  : {evaluacion.utilidad_relativa}/10")
    print(f"  justificacion      : {evaluacion.justificacion_breve}")
    print()
    print("El proveedor está listo. Puedes ejecutar:")
    print("  python src/run_fase2.py")


if __name__ == "__main__":
    main()
