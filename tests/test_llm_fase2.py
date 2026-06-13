"""
test_llm_fase2.py
=================
Pruebas unitarias para los componentes de la Fase 2.

Todos los tests son independientes de la API del LLM: se prueban los modelos
Pydantic, la lógica del evaluador con un cliente mock, y la persistencia.

Ejecutar:
    python -m pytest tests/test_llm_fase2.py -v
    # o sin pytest:
    python tests/test_llm_fase2.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from src.llm.models import EvaluacionCurso
from src.llm.prompts import construir_system_prompt, construir_user_prompt
from src.llm.cache import LLMCache
from src.llm.evaluator import evaluar_problema, guardar_problema_evaluado, resumen_evaluacion
from src.problem import Course, LearningPathProblem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_problem() -> LearningPathProblem:
    """Problema pequeño de 3 cursos para tests."""
    return LearningPathProblem(
        courses=[
            Course(id="A", titulo="Python", descripcion="Programacion basica.", duracion_horas=20, prerrequisitos=[]),
            Course(id="B", titulo="ML",     descripcion="Machine learning supervisado.", duracion_horas=40, prerrequisitos=["A"]),
            Course(id="C", titulo="NLP",    descripcion="Procesamiento de lenguaje natural.", duracion_horas=45, prerrequisitos=["B"]),
        ],
        t_max=100,
        instance_id="test_fase2",
    )


def make_evaluacion(curso_id: str = "A", utilidad: int = 8) -> EvaluacionCurso:
    return EvaluacionCurso(
        curso_id=curso_id,
        utilidad_relativa=utilidad,
        justificacion_breve="Curso muy relevante para el objetivo del usuario declarado.",
    )


# ---------------------------------------------------------------------------
# Tests: EvaluacionCurso (modelo Pydantic)
# ---------------------------------------------------------------------------

def test_evaluacion_valida():
    ev = make_evaluacion("CS_601", 9)
    assert ev.curso_id == "CS_601"
    assert ev.utilidad_relativa == 9
    assert len(ev.justificacion_breve) >= 20
    print("✓ test_evaluacion_valida")


def test_evaluacion_rango_minimo():
    ev = EvaluacionCurso(curso_id="X", utilidad_relativa=1, justificacion_breve="Curso irrelevante para el objetivo del usuario.")
    assert ev.utilidad_relativa == 1
    print("✓ test_evaluacion_rango_minimo")


def test_evaluacion_rechaza_utilidad_cero():
    try:
        EvaluacionCurso(curso_id="X", utilidad_relativa=0, justificacion_breve="Texto suficientemente largo para pasar validacion.")
        assert False, "Debería haber lanzado ValidationError"
    except ValidationError:
        pass
    print("✓ test_evaluacion_rechaza_utilidad_cero")


def test_evaluacion_rechaza_utilidad_once():
    try:
        EvaluacionCurso(curso_id="X", utilidad_relativa=11, justificacion_breve="Texto suficientemente largo para pasar validacion.")
        assert False, "Debería haber lanzado ValidationError"
    except ValidationError:
        pass
    print("✓ test_evaluacion_rechaza_utilidad_once")


def test_evaluacion_rechaza_justificacion_corta():
    try:
        EvaluacionCurso(curso_id="X", utilidad_relativa=5, justificacion_breve="Corta")
        assert False, "Debería haber lanzado ValidationError"
    except ValidationError:
        pass
    print("✓ test_evaluacion_rechaza_justificacion_corta")


def test_evaluacion_strip_curso_id():
    ev = EvaluacionCurso(curso_id="  CS_101  ", utilidad_relativa=7,
                         justificacion_breve="Curso con espacios en el ID que deben ser eliminados.")
    assert ev.curso_id == "CS_101"
    print("✓ test_evaluacion_strip_curso_id")


# ---------------------------------------------------------------------------
# Tests: Prompts
# ---------------------------------------------------------------------------

def test_system_prompt_no_vacio():
    sp = construir_system_prompt()
    assert len(sp) > 500, "System prompt demasiado corto"
    assert "Few-Shot" in sp or "Ejemplo" in sp, "System prompt debe incluir ejemplos"
    print(f"✓ test_system_prompt_no_vacio ({len(sp)} caracteres)")


def test_user_prompt_contiene_objetivo_y_curso():
    objetivo = "Aprender NLP y modelos de lenguaje."
    curso = {"id": "CS_501", "titulo": "NLP", "descripcion": "Curso de NLP avanzado."}
    up = construir_user_prompt(objetivo, curso)
    assert objetivo in up
    assert "CS_501" in up
    print("✓ test_user_prompt_contiene_objetivo_y_curso")


# ---------------------------------------------------------------------------
# Tests: LLMCache
# ---------------------------------------------------------------------------

def test_cache_get_miss():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    Path(tmp).write_text("{}")
    cache = LLMCache(tmp)
    assert cache.get("clave_inexistente") is None
    print("✓ test_cache_get_miss")


def test_cache_set_y_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = str(Path(tmpdir) / "cache.json")
        cache = LLMCache(cache_path)
        cache.set("prompt_prueba", '{"resultado": 42}')
        assert cache.get("prompt_prueba") == '{"resultado": 42}'
        assert len(cache) == 1
    print("✓ test_cache_set_y_get")


def test_cache_persiste_en_disco():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = str(Path(tmpdir) / "cache.json")
        cache1 = LLMCache(cache_path)
        cache1.set("k", "v")
        # Nueva instancia leyendo el mismo archivo
        cache2 = LLMCache(cache_path)
        assert cache2.get("k") == "v"
    print("✓ test_cache_persiste_en_disco")


# ---------------------------------------------------------------------------
# Tests: evaluar_problema con mock del LLMClient
# ---------------------------------------------------------------------------

def test_evaluar_problema_con_mock():
    """
    Simula una evaluación completa sin hacer llamadas reales a la API.
    El LLMClient se reemplaza por un mock que devuelve utilidades fijas.
    """
    problema = make_test_problem()
    objetivo = "Quiero especializarme en NLP y LLMs."

    # Mock del LLMClient: evaluar_curso devuelve puntuaciones fijas por ID
    utilidades_mock = {"A": 3, "B": 7, "C": 9}
    mock_client = MagicMock()
    mock_client.model = "mock-model"

    def evaluar_mock(curso: dict, objetivo: str) -> EvaluacionCurso:
        cid = curso["id"]
        return EvaluacionCurso(
            curso_id=cid,
            utilidad_relativa=utilidades_mock[cid],
            justificacion_breve=f"Justificacion mock para el curso {cid} respecto al objetivo.",
        )

    mock_client.evaluar_curso.side_effect = evaluar_mock

    resultado = evaluar_problema(
        problema=problema,
        objetivo_usuario=objetivo,
        llm_client=mock_client,
        delay_entre_llamadas=0.0,
    )

    # Verificar que las utilidades se aplicaron a los cursos
    courses_by_id = {c.id: c for c in resultado.courses}
    assert courses_by_id["A"].utilidad_relativa == 3
    assert courses_by_id["B"].utilidad_relativa == 7
    assert courses_by_id["C"].utilidad_relativa == 9

    # Verificar que las justificaciones se guardaron
    assert courses_by_id["C"].justificacion is not None

    # Verificar que las utilidades se reflejan en objective_value
    obj = resultado.objective_value(["A", "B"])
    assert obj == 3 + 7, f"Esperado 10, got {obj}"

    print("✓ test_evaluar_problema_con_mock")


def test_evaluar_problema_fallback_en_error():
    """
    Cuando el LLMClient devuelve None (error de API), el curso debe recibir
    la puntuación de fallback sin bloquear la evaluación de los demás.
    """
    problema = make_test_problem()

    mock_client = MagicMock()
    mock_client.model = "mock-model"
    mock_client.evaluar_curso.return_value = None  # Simula fallo total de API

    resultado = evaluar_problema(
        problema=problema,
        objetivo_usuario="Objetivo de prueba",
        llm_client=mock_client,
        delay_entre_llamadas=0.0,
        puntuacion_fallback=5,
    )

    # Todos los cursos deben tener la puntuación fallback
    for course in resultado.courses:
        assert course.utilidad_relativa == 5, f"[{course.id}] esperado 5, got {course.utilidad_relativa}"

    print("✓ test_evaluar_problema_fallback_en_error")


def test_guardar_y_cargar_problema_evaluado():
    """
    Verifica que el problema evaluado se serializa correctamente a JSON
    y que los campos utilidad_relativa están presentes en el archivo.
    """
    problema = make_test_problem()
    # Asignar utilidades manualmente
    for i, course in enumerate(problema.courses, start=1):
        course.utilidad_relativa = i * 3
        course.justificacion = f"Justificacion de prueba para {course.id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta = guardar_problema_evaluado(problema, directorio=tmpdir)
        assert ruta.exists()

        with ruta.open(encoding="utf-8") as f:
            datos = json.load(f)

        assert datos["instance_id"] == "test_fase2"
        assert len(datos["cursos"]) == 3
        assert datos["cursos"][0]["utilidad_relativa"] == 3
        assert datos["cursos"][1]["utilidad_relativa"] == 6
        assert datos["cursos"][2]["utilidad_relativa"] == 9

    print("✓ test_guardar_y_cargar_problema_evaluado")


def test_resumen_evaluacion():
    """Verifica que resumen_evaluacion genera texto no vacío con los datos correctos."""
    problema = make_test_problem()
    for i, course in enumerate(problema.courses, start=1):
        course.utilidad_relativa = i * 2
        course.justificacion = f"Razon {i} para incluir este curso en la ruta."

    resumen = resumen_evaluacion(problema, top_n=2)
    assert "test_fase2" in resumen
    assert "NLP" in resumen  # El curso C (mayor utilidad) debe aparecer primero
    print("✓ test_resumen_evaluacion")


# ---------------------------------------------------------------------------
# Runner manual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # EvaluacionCurso
        test_evaluacion_valida,
        test_evaluacion_rango_minimo,
        test_evaluacion_rechaza_utilidad_cero,
        test_evaluacion_rechaza_utilidad_once,
        test_evaluacion_rechaza_justificacion_corta,
        test_evaluacion_strip_curso_id,
        # Prompts
        test_system_prompt_no_vacio,
        test_user_prompt_contiene_objetivo_y_curso,
        # Cache
        test_cache_get_miss,
        test_cache_set_y_get,
        test_cache_persiste_en_disco,
        # Evaluator
        test_evaluar_problema_con_mock,
        test_evaluar_problema_fallback_en_error,
        test_guardar_y_cargar_problema_evaluado,
        test_resumen_evaluacion,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"✗ {fn.__name__}: {e}")
            failed += 1

    print(f"\n{'─' * 50}")
    print(f"Resultados: {passed} pasados, {failed} fallidos de {len(tests)} tests")
