"""
client.py  (src/llm/client.py)
================================
Wrapper multi-proveedor con fallback automático Gemini → Groq.

Comportamiento de fallback:
  - Proveedor primario: LLM_PROVIDER (default: gemini).
  - Proveedor de fallback: LLM_FALLBACK_PROVIDER (default: groq si primario es gemini).
  - Si el primario agota su cuota o no está disponible, el cliente cambia
    automáticamente al fallback y permanece en él el resto de la sesión.
  - Errores de autenticación (clave inválida) NO disparan el fallback —
    son errores de configuración que deben resolverse en .env.

Proveedores soportados:
  ┌──────────────┬─────────────────────────────┬─────────────────────────────────┐
  │ LLM_PROVIDER │ Modelo por defecto          │ Variable de clave               │
  ├──────────────┼─────────────────────────────┼─────────────────────────────────┤
  │ gemini       │ gemini-2.5-flash            │ GEMINI_API_KEY                  │
  │ groq         │ llama-3.3-70b-versatile     │ OPENAI_API_KEY (gsk_...)        │
  │ openai       │ gpt-4o-mini                 │ OPENAI_API_KEY (sk-...)         │
  └──────────────┴─────────────────────────────┴─────────────────────────────────┘

Variables de entorno relevantes:
  LLM_PROVIDER           proveedor activo            (default: gemini)
  LLM_FALLBACK_PROVIDER  proveedor de respaldo       (default: groq)
  LLM_MODEL              modelo del proveedor primario
  LLM_FALLBACK_MODEL     modelo del proveedor de fallback
  GEMINI_API_KEY         clave Google AI Studio
  OPENAI_API_KEY         clave OpenAI o Groq (gsk_...)
  OPENAI_BASE_URL        base URL alternativa (solo Groq)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from .prompts import construir_system_prompt, construir_user_prompt
from .models import EvaluacionCurso

load_dotenv()
logger = logging.getLogger(__name__)

# ── Cache del system prompt (invariante por sesión) ──────────────────────────
_SYSTEM_PROMPT: str = construir_system_prompt()

# ── Modelos por defecto de cada proveedor ────────────────────────────────────
_DEFAULT_MODEL: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}


# ─────────────────────────────────────────────────────────────────────────────
# Clasificación de errores: ¿dispara fallback o no?
# ─────────────────────────────────────────────────────────────────────────────

def _is_quota_error(exc: Exception) -> bool:
    """
    Devuelve True si el error indica cuota agotada o servicio no disponible
    —situaciones en las que tiene sentido cambiar al proveedor de respaldo.

    Devuelve False para errores de autenticación o de payload, que son
    problemas de configuración que no se resuelven cambiando de proveedor.
    """
    # ── Errores de Google Gemini ─────────────────────────────────────────────
    try:
        from google.api_core.exceptions import (
            ResourceExhausted,   # 429 — cuota diaria agotada
            ServiceUnavailable,  # 503 — servicio caído
            DeadlineExceeded,    # timeout tras reintentos
        )
        if isinstance(exc, (ResourceExhausted, ServiceUnavailable, DeadlineExceeded)):
            return True
    except ImportError:
        pass

    # ── Errores de OpenAI / Groq ─────────────────────────────────────────────
    try:
        from openai import RateLimitError, APIConnectionError, AuthenticationError
        if isinstance(exc, AuthenticationError):
            return False        # clave inválida → no hacer fallback
        if isinstance(exc, (RateLimitError, APIConnectionError)):
            return True
    except ImportError:
        pass

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend: encapsula un proveedor individual
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Backend:
    """Representa un proveedor LLM inicializado y listo para llamar."""
    name:    str
    model:   str
    _call_fn: Callable[[str], str] = field(repr=False)

    def call(self, user_prompt: str) -> str:
        return self._call_fn(user_prompt)


# ── Helpers de llamada con reintentos por proveedor ──────────────────────────

def _retry_gemini(fn):
    """Reintentos solo en errores transitorios de Gemini."""
    try:
        from google.api_core.exceptions import ServiceUnavailable, DeadlineExceeded
        recoverable = (ServiceUnavailable, DeadlineExceeded)
    except ImportError:
        recoverable = (Exception,)

    return retry(
        retry=retry_if_exception_type(recoverable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(fn)


def _retry_openai(fn):
    """Reintentos solo en errores transitorios de OpenAI/Groq."""
    try:
        from openai import APIConnectionError
        recoverable = (APIConnectionError,)
    except ImportError:
        recoverable = (Exception,)

    return retry(
        retry=retry_if_exception_type(recoverable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(fn)


def _build_gemini_backend(model: str) -> _Backend:
    """Construye el backend de Google Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY no configurada. "
            "Genera una clave en https://aistudio.google.com/"
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    @_retry_gemini
    def _call(user_prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=350,
            ),
        )
        return response.text or ""

    return _Backend(name="gemini", model=model, _call_fn=_call)


def _build_openai_backend(provider_name: str, model: str) -> _Backend:
    """Construye el backend de OpenAI o Groq (API compatible con OpenAI)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        hint = "gsk_... (Groq)" if provider_name == "groq" else "sk-... (OpenAI)"
        raise EnvironmentError(
            f"OPENAI_API_KEY no configurada para '{provider_name}'. "
            f"Valor esperado: {hint}"
        )

    from openai import OpenAI

    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    @_retry_openai
    def _call(user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=350,
        )
        return response.choices[0].message.content or ""

    return _Backend(name=provider_name, model=model, _call_fn=_call)


def _build_backend(provider: str, model: Optional[str] = None) -> _Backend:
    """Factory: construye el backend correcto según el nombre del proveedor."""
    resolved_model = model or _DEFAULT_MODEL.get(provider)
    if not resolved_model:
        raise ValueError(f"Proveedor desconocido: '{provider}'")

    if provider == "gemini":
        return _build_gemini_backend(resolved_model)
    elif provider in ("groq", "openai"):
        return _build_openai_backend(provider, resolved_model)
    else:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' no reconocido. "
            "Valores válidos: 'gemini', 'groq', 'openai'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLMClient — interfaz pública con fallback automático
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Cliente LLM multi-proveedor con fallback automático Gemini → Groq.

    Comportamiento:
      - Usa el proveedor primario (LLM_PROVIDER) mientras esté disponible.
      - Al detectar cuota agotada o servicio no disponible, cambia
        automáticamente al proveedor de fallback (LLM_FALLBACK_PROVIDER)
        y permanece en él el resto de la sesión.
      - El cambio se loguea claramente para que quede en el informe.

    Uso:
        client = LLMClient()
        evaluacion = client.evaluar_curso(curso_dict, objetivo_str)
    """

    def __init__(self) -> None:
        primary_name    = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        primary_model   = os.getenv("LLM_MODEL")

        # Fallback por defecto: groq si el primario es gemini, sino None
        default_fallback = "groq" if primary_name == "gemini" else None
        fallback_name   = os.getenv("LLM_FALLBACK_PROVIDER", default_fallback or "").lower().strip() or None
        fallback_model  = os.getenv("LLM_FALLBACK_MODEL")

        # Construir cadena de proveedores
        self._chain: List[_Backend] = []

        primary = _build_backend(primary_name, primary_model)
        self._chain.append(primary)
        logger.info("Proveedor primario   : %s | modelo: %s", primary.name, primary.model)

        if fallback_name and fallback_name != primary_name:
            try:
                fallback = _build_backend(fallback_name, fallback_model)
                self._chain.append(fallback)
                logger.info("Proveedor de fallback: %s | modelo: %s", fallback.name, fallback.model)
            except EnvironmentError as e:
                # Fallback no configurado → solo advertencia, no error fatal
                logger.warning(
                    "Fallback '%s' no disponible (falta clave): %s. "
                    "El sistema usará solo '%s'.",
                    fallback_name, e, primary_name,
                )

        self._active_idx: int = 0   # índice del proveedor activo en _chain

    # ── Propiedades de estado ─────────────────────────────────────────────────

    @property
    def active_backend(self) -> _Backend:
        return self._chain[self._active_idx]

    @property
    def provider(self) -> str:
        return self.active_backend.name

    @property
    def model(self) -> str:
        return self.active_backend.model

    @property
    def using_fallback(self) -> bool:
        return self._active_idx > 0

    # ── Llamada interna con lógica de fallback ────────────────────────────────

    def _call(self, user_prompt: str) -> str:
        """
        Llama al proveedor activo. Si falla por cuota, intenta el siguiente
        en la cadena y fija ese como activo para el resto de la sesión.
        """
        for idx in range(self._active_idx, len(self._chain)):
            backend = self._chain[idx]
            try:
                result = backend.call(user_prompt)

                # Si tuvimos que avanzar en la cadena, fijar el nuevo activo
                if idx != self._active_idx:
                    prev = self._chain[self._active_idx].name
                    self._active_idx = idx
                    logger.warning(
                        "⚡ Fallback activado: '%s' → '%s' (cuota agotada). "
                        "Se usará '%s' para el resto de la sesión.",
                        prev, backend.name, backend.name,
                    )
                return result

            except Exception as exc:
                if _is_quota_error(exc) and idx + 1 < len(self._chain):
                    logger.warning(
                        "Cuota agotada en '%s' (%s). Probando '%s'…",
                        backend.name,
                        type(exc).__name__,
                        self._chain[idx + 1].name,
                    )
                    continue   # probar siguiente en la cadena
                raise          # error no recuperable o sin más alternativas

        # Nunca debería llegar aquí, pero por completitud:
        raise RuntimeError("Todos los proveedores LLM fallaron.")

    # ── API pública ───────────────────────────────────────────────────────────

    def evaluar_curso(
        self,
        curso: dict,
        objetivo_usuario: str,
    ) -> Optional[EvaluacionCurso]:
        """
        Evalúa semánticamente un curso respecto al objetivo del usuario.

        Flujo:
          1. Construye el user prompt.
          2. Llama al proveedor activo (con fallback automático si aplica).
          3. Parsea el JSON de respuesta.
          4. Valida la estructura con Pydantic (EvaluacionCurso).

        Returns:
            EvaluacionCurso si tuvo éxito, None en caso de error irrecuperable.
        """
        from pydantic import ValidationError

        curso_id = curso.get("id", "DESCONOCIDO")
        user_prompt = construir_user_prompt(objetivo_usuario, curso)

        try:
            raw = self._call(user_prompt)
            datos = json.loads(raw)
            datos.setdefault("curso_id", curso_id)
            evaluacion = EvaluacionCurso.model_validate(datos)

            logger.info(
                "✓ [%s] u=%d/10 [%s] | %s",
                evaluacion.curso_id,
                evaluacion.utilidad_relativa,
                self.provider,
                (evaluacion.justificacion_breve[:70] + "…")
                if len(evaluacion.justificacion_breve) > 70
                else evaluacion.justificacion_breve,
            )
            return evaluacion

        except json.JSONDecodeError as e:
            logger.error("✗ [%s] Respuesta no es JSON válido: %s", curso_id, e)
        except ValidationError as e:
            logger.error("✗ [%s] Validación Pydantic fallida:\n%s", curso_id, e)
        except Exception as e:
            logger.error("✗ [%s] Error irrecuperable (%s): %s", curso_id, type(e).__name__, e)

        return None

    def __repr__(self) -> str:
        fallback_info = (
            f" → fallback={self._chain[1].name}" if len(self._chain) > 1 else ""
        )
        return (
            f"LLMClient(active={self.provider!r}, model={self.model!r}"
            f"{fallback_info})"
        )
