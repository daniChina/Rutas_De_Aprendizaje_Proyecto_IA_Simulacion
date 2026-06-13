"""
test_fallback.py  (tests/test_fallback.py)
==========================================
Pruebas unitarias para la lógica de fallback automático del LLMClient.

Todos los tests usan mocks — no requieren API keys reales.

Ejecutar:
    python tests/test_fallback.py
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de stub para aislar el módulo de dependencias externas
# ─────────────────────────────────────────────────────────────────────────────

RESPUESTA_VALIDA = json.dumps({
    "curso_id": "CS_101",
    "utilidad_relativa": 8,
    "justificacion_breve": "El curso es muy relevante para el objetivo del usuario.",
})

def _make_quota_error_gemini():
    """Crea una excepción ResourceExhausted de Google (cuota agotada)."""
    from google.api_core.exceptions import ResourceExhausted
    return ResourceExhausted("Cuota diaria agotada — 429")

def _make_rate_limit_error_openai():
    """Crea una excepción RateLimitError de OpenAI/Groq."""
    from openai import RateLimitError
    # RateLimitError requiere (message, response, body)
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    return RateLimitError("rate limit", response=mock_resp, body={})

def _make_auth_error_openai():
    """Crea una excepción AuthenticationError (NO debe disparar fallback)."""
    from openai import AuthenticationError
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    return AuthenticationError("invalid key", response=mock_resp, body={})


# ─────────────────────────────────────────────────────────────────────────────
# Tests de _is_quota_error
# ─────────────────────────────────────────────────────────────────────────────

def test_quota_error_gemini_resource_exhausted():
    from src.llm.client import _is_quota_error
    assert _is_quota_error(_make_quota_error_gemini()) is True
    print("✓ test_quota_error_gemini_resource_exhausted")

def test_quota_error_openai_rate_limit():
    from src.llm.client import _is_quota_error
    assert _is_quota_error(_make_rate_limit_error_openai()) is True
    print("✓ test_quota_error_openai_rate_limit")

def test_auth_error_no_es_quota():
    """AuthenticationError NO debe disparar fallback — es error de config."""
    from src.llm.client import _is_quota_error
    assert _is_quota_error(_make_auth_error_openai()) is False
    print("✓ test_auth_error_no_es_quota")

def test_error_generico_no_es_quota():
    from src.llm.client import _is_quota_error
    assert _is_quota_error(ValueError("error genérico")) is False
    print("✓ test_error_generico_no_es_quota")


# ─────────────────────────────────────────────────────────────────────────────
# Tests de la lógica de fallback en LLMClient._call
# ─────────────────────────────────────────────────────────────────────────────

def _make_client_with_mocks(primary_fails_with=None, fallback_response=RESPUESTA_VALIDA):
    """
    Construye un LLMClient con backends mockeados para tests.

    Args:
        primary_fails_with: Excepción que lanzará el backend primario.
                            None → el primario responde con RESPUESTA_VALIDA.
        fallback_response:  Respuesta del backend de fallback.
    """
    from src.llm.client import _Backend, LLMClient

    # Backend primario (Gemini mockeado)
    if primary_fails_with:
        primary_fn = MagicMock(side_effect=primary_fails_with)
    else:
        primary_fn = MagicMock(return_value=RESPUESTA_VALIDA)
    primary = _Backend(name="gemini", model="gemini-2.5-flash", _call_fn=primary_fn)

    # Backend de fallback (Groq mockeado)
    fallback_fn = MagicMock(return_value=fallback_response)
    fallback = _Backend(name="groq", model="llama-3.3-70b-versatile", _call_fn=fallback_fn)

    # Construir cliente directamente sin pasar por __init__ (evita leer .env)
    client = object.__new__(LLMClient)
    client._chain = [primary, fallback]
    client._active_idx = 0
    return client, primary_fn, fallback_fn


def test_primario_ok_no_usa_fallback():
    """Si el primario responde bien, el fallback nunca se llama."""
    client, primary_fn, fallback_fn = _make_client_with_mocks()
    result = client._call("prompt de prueba")
    assert result == RESPUESTA_VALIDA
    primary_fn.assert_called_once()
    fallback_fn.assert_not_called()
    assert client._active_idx == 0
    assert not client.using_fallback
    print("✓ test_primario_ok_no_usa_fallback")


def test_fallback_activa_en_cuota_gemini():
    """Cuota agotada en Gemini → llama al fallback Groq automáticamente."""
    client, primary_fn, fallback_fn = _make_client_with_mocks(
        primary_fails_with=_make_quota_error_gemini()
    )
    result = client._call("prompt de prueba")
    assert result == RESPUESTA_VALIDA
    primary_fn.assert_called_once()
    fallback_fn.assert_called_once()
    assert client._active_idx == 1      # se fijó en fallback
    assert client.using_fallback
    assert client.provider == "groq"
    print("✓ test_fallback_activa_en_cuota_gemini")


def test_fallback_permanece_resto_sesion():
    """Una vez activado el fallback, las llamadas siguientes van directo a él."""
    client, primary_fn, fallback_fn = _make_client_with_mocks(
        primary_fails_with=_make_quota_error_gemini()
    )
    # Primera llamada → activa el fallback
    client._call("prompt 1")
    assert client._active_idx == 1

    # Resetear contadores para verificar segunda llamada
    primary_fn.reset_mock()
    fallback_fn.reset_mock()

    # Segunda llamada → debe ir directamente al fallback, no intentar el primario
    client._call("prompt 2")
    primary_fn.assert_not_called()
    fallback_fn.assert_called_once()
    print("✓ test_fallback_permanece_resto_sesion")


def test_auth_error_no_activa_fallback():
    """Error de autenticación (clave inválida) NO debe disparar el fallback."""
    client, primary_fn, fallback_fn = _make_client_with_mocks(
        primary_fails_with=_make_auth_error_openai()
    )
    try:
        client._call("prompt de prueba")
        assert False, "Debería haber propagado la excepción"
    except Exception as exc:
        from openai import AuthenticationError
        assert isinstance(exc, AuthenticationError)

    fallback_fn.assert_not_called()
    assert client._active_idx == 0     # no se cambió de proveedor
    print("✓ test_auth_error_no_activa_fallback")


def test_sin_fallback_configurado_propaga_error():
    """Si no hay fallback en la cadena, el error se propaga normalmente."""
    from src.llm.client import _Backend, LLMClient

    primary_fn = MagicMock(side_effect=_make_quota_error_gemini())
    primary = _Backend(name="gemini", model="gemini-2.5-flash", _call_fn=primary_fn)

    client = object.__new__(LLMClient)
    client._chain = [primary]          # sin fallback
    client._active_idx = 0

    try:
        client._call("prompt de prueba")
        assert False, "Debería haber propagado la excepción"
    except Exception:
        pass  # error esperado

    assert client._active_idx == 0
    print("✓ test_sin_fallback_configurado_propaga_error")


def test_ambos_fallan_propaga_ultimo_error():
    """Si primario Y fallback fallan, se propaga el error del fallback."""
    from src.llm.client import _Backend, LLMClient

    quota_err = _make_quota_error_gemini()
    rate_err  = _make_rate_limit_error_openai()

    primary_fn  = MagicMock(side_effect=quota_err)
    fallback_fn = MagicMock(side_effect=rate_err)

    primary  = _Backend("gemini", "gemini-2.5-flash",          _call_fn=primary_fn)
    fallback = _Backend("groq",   "llama-3.3-70b-versatile",   _call_fn=fallback_fn)

    client = object.__new__(LLMClient)
    client._chain = [primary, fallback]
    client._active_idx = 0

    try:
        client._call("prompt de prueba")
        assert False, "Debería haber propagado la excepción"
    except Exception as exc:
        from openai import RateLimitError
        assert isinstance(exc, RateLimitError), f"Error inesperado: {exc}"

    print("✓ test_ambos_fallan_propaga_ultimo_error")


def test_provider_y_model_reflejan_activo():
    """Las propiedades .provider y .model muestran el proveedor actualmente activo."""
    client, _, _ = _make_client_with_mocks(
        primary_fails_with=_make_quota_error_gemini()
    )
    assert client.provider == "gemini"  # antes del fallback
    client._call("prompt")
    assert client.provider == "groq"    # después del fallback
    assert client.model == "llama-3.3-70b-versatile"
    print("✓ test_provider_y_model_reflejan_activo")


# ─────────────────────────────────────────────────────────────────────────────
# Test de integración: evaluar_curso con fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluar_curso_usa_fallback_transparentemente():
    """evaluar_curso devuelve EvaluacionCurso válida aunque el primario falle."""
    # Necesitamos que prompts.py y models.py existan
    # Usamos mocks para _call directamente
    from src.llm.client import _Backend, LLMClient

    primary_fn  = MagicMock(side_effect=_make_quota_error_gemini())
    fallback_fn = MagicMock(return_value=RESPUESTA_VALIDA)

    primary  = _Backend("gemini", "gemini-2.5-flash",        _call_fn=primary_fn)
    fallback = _Backend("groq",   "llama-3.3-70b-versatile", _call_fn=fallback_fn)

    client = object.__new__(LLMClient)
    client._chain = [primary, fallback]
    client._active_idx = 0

    # Mockear construir_user_prompt para no depender de prompts.py
    with patch("src.llm.client.construir_user_prompt", return_value="prompt"):
        curso = {"id": "CS_101", "titulo": "Python", "descripcion": "Curso base."}
        result = client.evaluar_curso(curso, "Aprender ML")

    assert result is not None
    assert result.curso_id == "CS_101"
    assert result.utilidad_relativa == 8
    assert client.using_fallback
    print("✓ test_evaluar_curso_usa_fallback_transparentemente")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_quota_error_gemini_resource_exhausted,
        test_quota_error_openai_rate_limit,
        test_auth_error_no_es_quota,
        test_error_generico_no_es_quota,
        test_primario_ok_no_usa_fallback,
        test_fallback_activa_en_cuota_gemini,
        test_fallback_permanece_resto_sesion,
        test_auth_error_no_activa_fallback,
        test_sin_fallback_configurado_propaga_error,
        test_ambos_fallan_propaga_ultimo_error,
        test_provider_y_model_reflejan_activo,
        test_evaluar_curso_usa_fallback_transparentemente,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"✗ {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'─'*50}")
    print(f"Resultados: {passed} pasados, {failed} fallidos de {len(tests)} tests")
