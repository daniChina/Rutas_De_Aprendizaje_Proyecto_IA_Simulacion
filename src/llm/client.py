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

import hashlib
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

from .cache import LLMCache
from .prompts import construir_system_prompt, construir_user_prompt
from .models import EvaluacionCurso

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
logger = logging.getLogger(__name__)

# ── Cache del system prompt (invariante por sesión) ──────────────────────────
_SYSTEM_PROMPT: str = construir_system_prompt()

# ── Modelos por defecto de cada proveedor ────────────────────────────────────
_DEFAULT_MODEL: dict[str, str] = {
    "groq":   "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
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
            ResourceExhausted,    # 429 — cuota diaria agotada
            ServiceUnavailable,   # 503 — servicio caído
            DeadlineExceeded,     # timeout tras reintentos
            InternalServerError,  # 500 — error interno del servidor
        )
        if isinstance(exc, (ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError)):
            return True
    except ImportError:
        pass

    # ── Detección genérica por código HTTP (cubre ServerError y similares) ───
    if hasattr(exc, "code") and getattr(exc, "code") in (429, 500, 503):
        return True
    # Algunos SDKs exponen el código en .status_code en lugar de .code
    if hasattr(exc, "status_code") and getattr(exc, "status_code") in (429, 500, 503):
        return True
    # Por si el mensaje contiene el código (último recurso)
    msg = str(exc).lower()
    if "503" in msg or "unavailable" in msg or "rate limit" in msg:
        return True

    # ── Errores de OpenAI / Groq ─────────────────────────────────────────────
    try:
        from openai import RateLimitError, APIConnectionError, APIStatusError, AuthenticationError
        if isinstance(exc, AuthenticationError):
            return False  # clave inválida → no hacer fallback
        if isinstance(exc, (RateLimitError, APIConnectionError)):
            return True
        # APIStatusError cubre 500/503 de OpenAI y Groq
        if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 503):
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
        texto = response.text or ""
    
        if not texto.strip():
            from google.api_core.exceptions import ServiceUnavailable
            raise ServiceUnavailable("Gemini devolvió respuesta vacía")
        return texto

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
            max_tokens=150,
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
# ── Utilidad: limpiar bloques markdown de la respuesta ───────
def _limpiar_json(texto: str) -> str:
    """Elimina bloques ```json ... ``` que algunos modelos añaden."""
    texto = texto.strip()
    if texto.startswith("```"):
        lineas = texto.splitlines()
        lineas = [l for l in lineas if not l.strip().startswith("```")]
        texto = "\n".join(lineas).strip()
    return texto

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
        client = LLMClient(cache_path=".llm_cache.json")
        evaluacion = client.evaluar_curso(curso_dict, objetivo_str)
    """

    def __init__(self, cache_path: Optional[str] = ".llm_cache.json") -> None:
        primary_name    = os.getenv("LLM_PROVIDER", "groq").lower().strip()
        primary_model   = os.getenv("LLM_MODEL")
        print(f"LLM_PROVIDER: '{primary_name}' | LLM_MODEL: '{primary_model or _DEFAULT_MODEL.get(primary_name, 'N/A')}'")
        

        # Fallback por defecto: groq si el primario es gemini, sino None
        default_fallback = "gemini" if primary_name == "groq" else None
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
        self._cache: Optional[LLMCache] = LLMCache(cache_path) if cache_path else None
        if self._cache is not None:
            stats = self._cache.stats()
            logger.info(
                "Caché LLM activa: %s (%d entradas, %.1f KB)",
                stats["path"], stats["entries"], stats["size_kb"],
            )

    def _cache_key(self, user_prompt: str) -> str:
        """Construye una clave determinista a partir del system prompt y el user prompt.
        El user prompt ya contiene el objetivo del usuario y los datos del curso,
        por lo que dos evaluaciones del mismo curso con distinto objetivo producen
        claves distintas (comportamiento correcto).
        """
        digest = hashlib.sha256(
            (f"{_SYSTEM_PROMPT}\n{user_prompt}").encode("utf-8")
        ).hexdigest()
        return digest

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

        def _parse_response(raw_text: str) -> EvaluacionCurso:
            raw_limpio = _limpiar_json(raw_text)
            if not raw_limpio:
                raise ValueError("El LLM devolvió una respuesta vacía")
            datos = json.loads(raw_limpio)
            datos.setdefault("curso_id", curso_id)
            return EvaluacionCurso.model_validate(datos)

        def _evaluate() -> EvaluacionCurso:
            cache_key = self._cache_key(user_prompt) if self._cache else None
            if cache_key is not None:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.info("♫ Cache hit para curso %s", curso_id)
                    try:
                        return _parse_response(cached)
                    except (json.JSONDecodeError, ValidationError, ValueError) as e:
                        logger.warning(
                            "Cache inválida para curso %s: %s. Ignorando entrada cacheada.",
                            curso_id,
                            e,
                        )

            raw = self._call(user_prompt)
            evaluacion = _parse_response(raw)

            if cache_key is not None:
                self._cache.set(cache_key, raw)
                logger.info("♫ Cache guardada para curso %s", curso_id)

            return evaluacion

        try:
            evaluacion = _evaluate()
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

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            if self._active_idx + 1 < len(self._chain):
                logger.warning(
                    "⚠ [%s] Respuesta inválida desde '%s': %s. Intentando fallback a '%s'.",
                    curso_id,
                    self.provider,
                    e,
                    self._chain[self._active_idx + 1].name,
                )
                self._active_idx += 1
                try:
                    evaluacion = _evaluate()
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
                except (json.JSONDecodeError, ValidationError, ValueError, Exception) as fallback_exc:
                    logger.error(
                        "✗ [%s] El fallback '%s' también falló: %s",
                        curso_id,
                        self.provider,
                        fallback_exc,
                    )
                    return None

            if isinstance(e, json.JSONDecodeError):
                logger.error("✗ [%s] Respuesta no es JSON válido: %s", curso_id, e)
            elif isinstance(e, ValidationError):
                logger.error("✗ [%s] Validación Pydantic fallida:\n%s", curso_id, e)
            else:
                logger.error("✗ [%s] El LLM devolvió una respuesta vacía", curso_id)

        except Exception as e:
            logger.error("✗ [%s] Error irrecuperable (%s): %s", curso_id, type(e).__name__, e)

        return None

    def __repr__(self) -> str:
        fallback_info = (
            f" → fallback={self._chain[1].name}" if len(self._chain) > 1 else ""
        )
        cache_info = (
            f", cache={self._cache._path}" if self._cache else ", cache=off"
        )
        return (
            f"LLMClient(active={self.provider!r}, model={self.model!r}"
            f"{fallback_info}{cache_info})"
        )
