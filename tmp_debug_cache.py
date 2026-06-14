from src.llm.client import LLMClient, _Backend
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

raw_response = json.dumps({
    "curso_id": "CACHE_TEST",
    "utilidad_relativa": 8,
    "justificacion_breve": "Esta justificacion es suficientemente larga para pasar la validacion.",
})

with tempfile.TemporaryDirectory() as tmpdir:
    cache_path = str(Path(tmpdir) / "cache.json")
    with patch("src.llm.client._build_backend") as mock_build_backend, patch.object(LLMClient, "_call", return_value=raw_response) as mock_call:
        mock_build_backend.return_value = _Backend(
            name="mock",
            model="mock-model",
            _call_fn=lambda prompt: raw_response,
        )
        client = LLMClient(cache_path=cache_path)
        print("cache object", client._cache)
        curso = {"id": "CACHE_TEST", "titulo": "Curso cache", "descripcion": "Descripcion."}

        evaluacion1 = client.evaluar_curso(curso, "Objetivo de prueba")
        print("first", mock_call.call_count)
        print("evaluacion1", evaluacion1)
        print("cache after first", client._cache._data if client._cache else None)
        evaluacion2 = client.evaluar_curso(curso, "Objetivo de prueba")
        print("second", mock_call.call_count)
        print("evaluacion2", evaluacion2)
        print("cache after second", client._cache._data if client._cache else None)
        print("cache file exists", Path(cache_path).exists())
