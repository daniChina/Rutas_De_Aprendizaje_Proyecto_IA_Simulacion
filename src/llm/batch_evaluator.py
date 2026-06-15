"""
batch_evaluator.py  (src/llm/batch_evaluator.py)
=================================================
Evaluador por lotes: evalúa TODOS los cursos en una sola llamada al LLM.

Problema que resuelve:
    El evaluador original (evaluator.py) hace una llamada por curso.
    Para la Instancia C (35 cursos) eso son ~64.000 tokens de input
    porque el system prompt (~1.300 tokens) se repite en cada llamada.

    Este módulo agrupa todos los cursos en un único user prompt y hace
    UNA sola llamada, reduciendo el consumo a ~6.000 tokens (~90% menos).

Cuándo usar cada evaluador:
    ┌────────────────────┬──────────────────┬────────────────────────┐
    │                    │ evaluator.py     │ batch_evaluator.py     │
    ├────────────────────┼──────────────────┼────────────────────────┤
    │ Instancias A, B    │ ✓ recomendado    │ también funciona       │
    │ Instancia C (35)   │ agota tokens     │ ✓ recomendado          │
    │ Cursos nuevos      │ ✓ (reutiliza     │ evalúa todo de nuevo   │
    │ (caché parcial)    │   caché)         │ (no aprovecha caché)   │
    │ Rate limit estricto│ ✓ (delay entre   │ una sola llamada       │
    │                    │   llamadas)      │                        │
    └────────────────────┴──────────────────┴────────────────────────┘

Estrategia interna:
    1. Filtra los cursos que YA tienen utilidad_relativa (evita re-evaluar).
    2. Agrupa los pendientes en lotes de BATCH_SIZE (default 35).
    3. Por cada lote: construye un user prompt con todos los cursos en JSON,
       hace UNA llamada al LLM y parsea el array de respuestas.
    4. Si la respuesta del lote es inválida, hace fallback al evaluador
       individual para ese lote (sin perder los ya evaluados).
    5. Actualiza course.utilidad_relativa y course.justificacion in-place.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from ..problem import Course, LearningPathProblem
from .client import LLMClient
from .models import EvaluacionCurso
from .evaluator import evaluar_problema   # fallback individual

logger = logging.getLogger(__name__)

# Número máximo de cursos por llamada.
# 35 cabe perfectamente en un solo lote (~6k tokens total).
# Si tienes instancias mayores, baja a 20 para mayor fiabilidad.
BATCH_SIZE = 35


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del prompt por lotes
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BATCH = """\
Eres un experto en diseño curricular y evaluación pedagógica.
Tu tarea es evaluar la relevancia semántica de una lista de cursos respecto \
al objetivo de aprendizaje declarado por el usuario.

## Escala de utilidad (1–10)
| Rango | Significado |
|-------|-------------|
| 9–10  | Esencial para el objetivo — habilidades o conocimientos nucleares. |
| 7–8   | Muy relevante — aporta competencias importantes aunque no centrales. |
| 5–6   | Moderadamente útil — fundamentos o habilidades complementarias. |
| 3–4   | Conexión débil o periférica con el objetivo. |
| 1–2   | Irrelevante — dominio distinto al objetivo declarado. |

## Formato de salida OBLIGATORIO
Responde ÚNICAMENTE con un array JSON válido. Sin texto adicional, sin \
bloques markdown. Cada elemento del array tiene exactamente estos campos:
  "curso_id"           — string, el ID exacto del curso tal como se proporcionó
  "utilidad_relativa"  — integer 1–10
  "justificacion_breve"— string, 1–2 oraciones máximo (≤ 120 caracteres)

El array debe tener exactamente tantos elementos como cursos recibiste, \
en el mismo orden.

Ejemplo de salida válida:
[
  {"curso_id": "CS_101", "utilidad_relativa": 8, "justificacion_breve": "Cubre los fundamentos esenciales para el objetivo."},
  {"curso_id": "CS_202", "utilidad_relativa": 3, "justificacion_breve": "Área periférica con escasa transferencia al objetivo."}
]
"""


def _construir_prompt_lote(
    cursos: list[Course],
    objetivo_usuario: str,
) -> str:
    """Construye el user prompt con todos los cursos del lote en JSON."""
    cursos_json = [
        {
            "id":          c.id,
            "titulo":      c.titulo,
            "descripcion": c.descripcion[:300],   # truncar descripciones largas
        }
        for c in cursos
    ]
    return (
        f"Objetivo del usuario:\n{objetivo_usuario.strip()}\n\n"
        f"Cursos a evaluar ({len(cursos)}):\n"
        f"{json.dumps(cursos_json, ensure_ascii=False, indent=2)}\n\n"
        "Devuelve el array JSON de evaluaciones."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parseo de la respuesta del lote
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_respuesta_lote(
    raw: str,
    cursos_lote: list[Course],
    puntuacion_fallback: int,
) -> list[Optional[EvaluacionCurso]]:
    """
    Parsea el array JSON devuelto por el LLM para un lote de cursos.
    Devuelve una lista paralela a cursos_lote: EvaluacionCurso o None.
    """
    from pydantic import ValidationError

    # Limpiar bloques markdown si el modelo los añade
    texto = raw.strip()
    if texto.startswith("```"):
        lineas = [l for l in texto.splitlines() if not l.strip().startswith("```")]
        texto = "\n".join(lineas).strip()

    try:
        datos = json.loads(texto)
    except json.JSONDecodeError as e:
        logger.error("Lote: respuesta no es JSON válido: %s\nRaw: %.200s", e, raw)
        return [None] * len(cursos_lote)

    if not isinstance(datos, list):
        # Algunos modelos envuelven en {"evaluaciones": [...]}
        if isinstance(datos, dict):
            for key in ("evaluaciones", "cursos", "results", "data"):
                if key in datos and isinstance(datos[key], list):
                    datos = datos[key]
                    break
        if not isinstance(datos, list):
            logger.error("Lote: respuesta no es un array. Tipo: %s", type(datos))
            return [None] * len(cursos_lote)

    # Construir índice por curso_id para emparejar con el orden original
    index_por_id: dict[str, dict] = {}
    for item in datos:
        if isinstance(item, dict) and "curso_id" in item:
            index_por_id[str(item["curso_id"])] = item

    resultados: list[Optional[EvaluacionCurso]] = []
    for course in cursos_lote:
        item = index_por_id.get(course.id)
        if item is None:
            # Buscar por posición si el LLM no respetó el ID
            pos = cursos_lote.index(course)
            item = datos[pos] if pos < len(datos) else None

        if item is None:
            logger.warning("Lote: no se encontró evaluación para curso %s", course.id)
            resultados.append(None)
            continue

        try:
            item.setdefault("curso_id", course.id)
            # Truncar justificacion si supera el límite de Pydantic (600 chars)
            if len(item.get("justificacion_breve", "")) > 600:
                item["justificacion_breve"] = item["justificacion_breve"][:597] + "..."
            # Rellenar justificacion si viene vacía o muy corta
            if len(item.get("justificacion_breve", "").strip()) < 20:
                item["justificacion_breve"] = (
                    f"Curso evaluado con utilidad {item.get('utilidad_relativa', puntuacion_fallback)}/10 "
                    "respecto al objetivo declarado."
                )
            ev = EvaluacionCurso.model_validate(item)
            resultados.append(ev)
        except (ValidationError, Exception) as e:
            logger.warning("Lote: validación fallida para %s: %s", course.id, e)
            resultados.append(None)

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# Evaluador por lotes principal
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_problema_batch(
    problema: LearningPathProblem,
    objetivo_usuario: str,
    *,
    llm_client: Optional[LLMClient] = None,
    batch_size: int = BATCH_SIZE,
    puntuacion_fallback: int = 5,
    fallback_a_individual: bool = True,
) -> LearningPathProblem:
    """
    Evalúa semánticamente todos los cursos del problema en lotes,
    reduciendo el consumo de tokens en ~90% respecto al evaluador individual.

    Diferencias con evaluar_problema() del evaluator.py:
      - Agrupa hasta `batch_size` cursos por llamada al LLM.
      - La justificacion_breve se limita a 120 caracteres (suficiente para
        el informe, y reduce los tokens de output significativamente).
      - Si un lote falla, hace fallback al evaluador individual curso a curso.
      - No usa la caché del LLMClient (el prompt por lotes es diferente al
        individual; mezclarlos contaminaría la caché).

    Args:
        problema:              Instancia con cursos a evaluar.
        objetivo_usuario:      Meta de aprendizaje en lenguaje natural.
        llm_client:            Cliente LLM ya construido (o se crea uno).
        batch_size:            Cursos por llamada (default 35, máx recomendado).
        puntuacion_fallback:   Utilidad asignada si el lote falla totalmente.
        fallback_a_individual: Si True, los cursos que fallen en batch se
                               re-intentan con el evaluador individual.

    Returns:
        El mismo LearningPathProblem con utilidad_relativa actualizada.
    """
    client = llm_client or LLMClient()   # sin caché para el batch

    # Separar cursos pendientes de los ya evaluados
    pendientes = [c for c in problema.courses if c.utilidad_relativa is None]
    ya_evaluados = len(problema.courses) - len(pendientes)

    logger.info("=" * 60)
    logger.info("Evaluación BATCH — instancia=%s", problema.instance_id)
    logger.info("Modelo LLM : %s", client.model)
    logger.info("Total      : %d cursos | %d ya evaluados | %d pendientes",
                len(problema.courses), ya_evaluados, len(pendientes))
    logger.info("Lotes      : %d cursos/lote → %d llamadas al LLM",
                batch_size, -(-len(pendientes) // batch_size))  # ceil
    logger.info("=" * 60)

    if not pendientes:
        logger.info("Todos los cursos ya tienen utilidad. Nada que hacer.")
        return problema

    # Dividir en lotes
    lotes = [pendientes[i:i + batch_size] for i in range(0, len(pendientes), batch_size)]
    exitosos_total = 0
    fallidos_total: list[str] = []

    for num_lote, lote in enumerate(lotes, start=1):
        logger.info("Lote %d/%d: %d cursos (%s … %s)",
                    num_lote, len(lotes), len(lote), lote[0].id, lote[-1].id)

        user_prompt = _construir_prompt_lote(lote, objetivo_usuario)

        # ── Sustituir temporalmente el system prompt del cliente ──────────────
        # El LLMClient usa _SYSTEM_PROMPT global (el individual).
        # Para batch necesitamos nuestro propio system prompt.
        # Lo hacemos parcheando el backend temporalmente.
        raw_response: Optional[str] = None
        try:
            raw_response = _llamar_con_system_prompt_batch(client, user_prompt)
        except Exception as exc:
            logger.error("Lote %d: llamada al LLM falló: %s", num_lote, exc)

        if raw_response:
            evaluaciones = _parsear_respuesta_lote(raw_response, lote, puntuacion_fallback)
        else:
            evaluaciones = [None] * len(lote)

        # Aplicar evaluaciones al problema
        fallidos_lote: list[Course] = []
        for course, ev in zip(lote, evaluaciones):
            if ev is not None:
                course.utilidad_relativa = ev.utilidad_relativa
                course.justificacion    = ev.justificacion_breve
                exitosos_total += 1
                logger.info("  ✓ [%s] u=%d/10", course.id, ev.utilidad_relativa)
            else:
                fallidos_lote.append(course)
                fallidos_total.append(course.id)

        # ── Fallback individual para cursos que fallaron en el lote ──────────
        if fallidos_lote and fallback_a_individual:
            logger.warning(
                "Lote %d: %d cursos fallidos → re-intentando individualmente: %s",
                num_lote, len(fallidos_lote), [c.id for c in fallidos_lote],
            )
            # Crear un sub-problema temporal solo con los fallidos
            from ..problem import LearningPathProblem as LPP
            sub = LPP(
                courses=fallidos_lote,
                t_max=problema.t_max,
                instance_id=f"{problema.instance_id}_retry",
            )
            sub = evaluar_problema(
                sub,
                objetivo_usuario,
                llm_client=client,
                delay_entre_llamadas=0.3,
                puntuacion_fallback=puntuacion_fallback,
            )
            # Los cambios se aplican in-place sobre los mismos objetos Course
            for c in fallidos_lote:
                if c.utilidad_relativa is not None:
                    exitosos_total += 1
                    fallidos_total.remove(c.id)

        # Pausa entre lotes (no entre cursos)
        if num_lote < len(lotes):
            time.sleep(1.0)

    logger.info("=" * 60)
    logger.info("Batch completado: %d/%d exitosos | fallidos: %s",
                exitosos_total, len(pendientes), fallidos_total or "ninguno")
    logger.info("=" * 60)

    # Asignar fallback a los que quedaron sin evaluar
    for course in problema.courses:
        if course.utilidad_relativa is None:
            course.utilidad_relativa = puntuacion_fallback
            course.justificacion = (
                f"[FALLBACK] Evaluación no disponible. "
                f"Puntuación neutral {puntuacion_fallback}/10."
            )

    return problema


# ─────────────────────────────────────────────────────────────────────────────
# Helper: llamada con system prompt de batch
# ─────────────────────────────────────────────────────────────────────────────

def _llamar_con_system_prompt_batch(client: LLMClient, user_prompt: str) -> str:
    """
    Realiza la llamada al LLM usando el system prompt de batch en lugar
    del system prompt individual que tiene configurado el cliente.

    Estrategia: llamamos directamente al backend activo construyendo
    el request manualmente, para no mezclar prompts en la caché.
    """
    backend = client.active_backend

    if backend.name == "gemini":
        return _llamar_gemini_batch(backend.model, user_prompt)
    elif backend.name in ("openai", "groq"):
        return _llamar_openai_batch(backend.model, user_prompt, client)
    else:
        raise ValueError(f"Backend no soportado para batch: {backend.name}")


def _llamar_gemini_batch(model: str, user_prompt: str) -> str:
    import os
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)

    response = gemini_client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT_BATCH,
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=4096,   # más tokens de output para el array completo
        ),
    )
    return response.text or ""


def _llamar_openai_batch(model: str, user_prompt: str, client: LLMClient) -> str:
    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    oa_client = OpenAI(**kwargs)

    response = oa_client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_BATCH},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""
