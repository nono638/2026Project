"""task-053 — tests for Ollama quantization tracking helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from experiment_utils import (
    get_ollama_model_details,
    write_experiment_metadata,
)


def _make_response(status_code: int, payload: dict | None = None) -> MagicMock:
    """Build a fake requests.Response with .ok, .status_code, .json(), .text."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.text = json.dumps(payload) if payload is not None else ""
    if payload is not None:
        resp.json = MagicMock(return_value=payload)
    else:
        resp.json = MagicMock(side_effect=ValueError("no body"))
    return resp


class TestGetOllamaModelDetails:
    """Cover happy path, unreachable host, 404, and missing-fields cases."""

    def test_success(self) -> None:
        body = {
            "digest": "sha256:abc123",
            "details": {
                "quantization_level": "Q4_K_M",
                "parameter_size": "4.0B",
                "family": "qwen3",
                "format": "gguf",
            },
        }
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(200, body)) as mock_post:
            out = get_ollama_model_details("qwen3:4b")
        assert out["tag"] == "qwen3:4b"
        assert out["digest"] == "sha256:abc123"
        assert out["quantization_level"] == "Q4_K_M"
        assert out["parameter_size"] == "4.0B"
        assert out["family"] == "qwen3"
        assert out["format"] == "gguf"
        assert out["captured_at"]  # ISO timestamp populated
        mock_post.assert_called_once()
        # Default host is localhost
        args, kwargs = mock_post.call_args
        assert "localhost:11434" in args[0]
        assert kwargs["json"] == {"model": "qwen3:4b", "verbose": False}

    def test_explicit_host_used(self) -> None:
        body = {"digest": "sha256:x", "details": {"quantization_level": "F16"}}
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(200, body)) as mock_post:
            get_ollama_model_details("qwen3:4b", host="http://gpu-pod:11434")
        url = mock_post.call_args[0][0]
        assert url == "http://gpu-pod:11434/api/show"

    def test_bare_host_gets_http_prefix(self) -> None:
        body = {"digest": "sha256:x", "details": {"quantization_level": "F16"}}
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(200, body)) as mock_post:
            get_ollama_model_details("qwen3:4b", host="gpu-pod:11434")
        url = mock_post.call_args[0][0]
        assert url.startswith("http://gpu-pod:11434/")

    def test_unreachable_host(self) -> None:
        with patch("experiment_utils.requests.post",
                   side_effect=requests.ConnectionError("boom")):
            out = get_ollama_model_details("qwen3:4b")
        assert out["tag"] == "qwen3:4b"
        assert out["captured_at"]
        for k in ("digest", "quantization_level", "parameter_size", "family", "format"):
            assert out[k] is None

    def test_404(self) -> None:
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(404, {"error": "model not found"})):
            out = get_ollama_model_details("does-not-exist:99b")
        assert out["tag"] == "does-not-exist:99b"
        assert out["quantization_level"] is None
        assert out["digest"] is None

    def test_missing_details_field(self) -> None:
        body = {"digest": "sha256:abc"}  # no details key at all
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(200, body)):
            out = get_ollama_model_details("qwen3:4b")
        assert out["digest"] == "sha256:abc"
        assert out["quantization_level"] is None
        assert out["family"] is None

    def test_partial_details_field(self) -> None:
        body = {"digest": "sha256:abc", "details": {"family": "qwen3"}}
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(200, body)):
            out = get_ollama_model_details("qwen3:4b")
        assert out["family"] == "qwen3"
        assert out["quantization_level"] is None

    def test_non_json_body(self) -> None:
        with patch("experiment_utils.requests.post",
                   return_value=_make_response(200, payload=None)):
            out = get_ollama_model_details("qwen3:4b")
        assert out["quantization_level"] is None
        assert out["captured_at"]


class TestWriteExperimentMetadataModelDetails:
    """Cover model_details kwarg present/absent in the metadata.json output."""

    def test_with_model_details(self, tmp_path: Path) -> None:
        details = {
            "qwen3:4b": {
                "tag": "qwen3:4b",
                "digest": "sha256:abc",
                "quantization_level": "Q4_K_M",
                "parameter_size": "4.0B",
                "family": "qwen3",
                "format": "gguf",
                "captured_at": "2026-05-06T12:00:00",
            }
        }
        write_experiment_metadata(
            output_dir=tmp_path,
            n_examples=10,
            config={"model": "qwen3:4b"},
            judges=[],
            model_details=details,
        )
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["model_details"] == details

    def test_without_model_details_key_absent(self, tmp_path: Path) -> None:
        write_experiment_metadata(
            output_dir=tmp_path,
            n_examples=5,
            config={"model": "qwen3:4b"},
            judges=[],
        )
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert "model_details" not in meta

    def test_empty_model_details_key_absent(self, tmp_path: Path) -> None:
        # Falsy dict — empty — should not produce an empty "model_details" field
        write_experiment_metadata(
            output_dir=tmp_path,
            n_examples=5,
            config={"model": "qwen3:4b"},
            judges=[],
            model_details={},
        )
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert "model_details" not in meta


class TestBackfillScript:
    """Cover idempotency and the backfill_note field."""

    def test_skips_when_model_details_present(self, tmp_path: Path) -> None:
        from backfill_quant_metadata import backfill_one

        meta = {
            "experiment": "x",
            "config": {"model": "qwen3:4b"},
            "model_details": {"qwen3:4b": {"quantization_level": "Q4_K_M"}},
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        with patch("backfill_quant_metadata.get_ollama_model_details") as mock_get:
            wrote = backfill_one(tmp_path)
        assert wrote is False
        mock_get.assert_not_called()
        # File unchanged
        loaded = json.loads((tmp_path / "metadata.json").read_text())
        assert loaded == meta

    def test_writes_model_details_and_note(self, tmp_path: Path) -> None:
        from backfill_quant_metadata import backfill_one

        meta = {
            "experiment": "experiment_0",
            "config": {"model": "qwen3:4b"},
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        fake_details = {
            "tag": "qwen3:4b",
            "quantization_level": "Q4_K_M",
            "digest": "sha256:abc",
            "parameter_size": "4.0B",
            "family": "qwen3",
            "format": "gguf",
            "captured_at": "2026-05-06T12:00:00",
        }
        with patch("backfill_quant_metadata.get_ollama_model_details",
                   return_value=fake_details) as mock_get:
            wrote = backfill_one(tmp_path)
        assert wrote is True
        mock_get.assert_called_once_with("qwen3:4b", host=None)

        loaded = json.loads((tmp_path / "metadata.json").read_text())
        assert loaded["model_details"]["qwen3:4b"]["quantization_level"] == "Q4_K_M"
        assert "backfill_note" in loaded
        assert "Backfilled retroactively" in loaded["backfill_note"]


class TestGenerateAnswerLlmQuantization:
    """Verify generate_answer copies quantization_level into the row dict."""

    def test_llm_quantization_from_model_details(self, tmp_path: Path) -> None:
        # Mock out the heavy strategy/retriever stack: generate_answer takes
        # arbitrary objects; we only need .text/.reference_answer on query/doc
        # and a strategy.run that returns an answer string.
        from experiment_utils import generate_answer

        fake_query = MagicMock()
        fake_query.text = "what is 2+2"
        fake_query.reference_answer = "4"
        fake_doc = MagicMock()
        fake_doc.text = "Two plus two equals four. The answer is 4."

        fake_chunker = MagicMock()
        fake_chunker.chunk = MagicMock(return_value=["Two plus two equals four. The answer is 4."])

        # Patch the Retriever construction inside experiment_utils so we
        # don't need a real embedder.
        fake_retriever = MagicMock()
        with patch("experiment_utils.Retriever", return_value=fake_retriever):
            fake_strategy = MagicMock()

            def _run(query, retriever, model, diagnostics=None):
                # Populate diagnostics like a real strategy would
                if diagnostics is not None:
                    diagnostics["context_sent_to_llm"] = "Two plus two equals four."
                    diagnostics["retrieved_chunks"] = [
                        {"text": "Two plus two equals four."},
                    ]
                    diagnostics["skipped_retrieval"] = False
                return "4"
            fake_strategy.run = _run

            row = generate_answer(
                strategy=fake_strategy,
                chunker=fake_chunker,
                embedder=MagicMock(),
                retrieval_mode="hybrid",
                query=fake_query,
                doc=fake_doc,
                model="qwen3:4b",
                model_details={
                    "tag": "qwen3:4b",
                    "quantization_level": "Q4_K_M",
                },
            )
        assert row["llm_quantization"] == "Q4_K_M"

    def test_llm_quantization_unknown_when_details_missing(self) -> None:
        from experiment_utils import generate_answer

        fake_query = MagicMock(); fake_query.text = "q"; fake_query.reference_answer = ""
        fake_doc = MagicMock(); fake_doc.text = "d"
        fake_chunker = MagicMock(); fake_chunker.chunk = MagicMock(return_value=["c"])

        with patch("experiment_utils.Retriever", return_value=MagicMock()):
            fake_strategy = MagicMock()
            fake_strategy.run = lambda *a, **k: "ans"
            row = generate_answer(
                strategy=fake_strategy,
                chunker=fake_chunker,
                embedder=MagicMock(),
                retrieval_mode="hybrid",
                query=fake_query,
                doc=fake_doc,
                model="qwen3:4b",
                model_details=None,
            )
        assert row["llm_quantization"] == "unknown"

    def test_llm_quantization_unknown_when_quant_field_none(self) -> None:
        from experiment_utils import generate_answer

        fake_query = MagicMock(); fake_query.text = "q"; fake_query.reference_answer = ""
        fake_doc = MagicMock(); fake_doc.text = "d"
        fake_chunker = MagicMock(); fake_chunker.chunk = MagicMock(return_value=["c"])

        with patch("experiment_utils.Retriever", return_value=MagicMock()):
            fake_strategy = MagicMock()
            fake_strategy.run = lambda *a, **k: "ans"
            row = generate_answer(
                strategy=fake_strategy,
                chunker=fake_chunker,
                embedder=MagicMock(),
                retrieval_mode="hybrid",
                query=fake_query,
                doc=fake_doc,
                model="qwen3:4b",
                model_details={"tag": "qwen3:4b", "quantization_level": None},
            )
        assert row["llm_quantization"] == "unknown"
