"""Tests for Ollama Modelfile and GGUF quantization export."""

import json
from pathlib import Path
import pytest

from mergestudio.export.ollama import (
    generate_gguf_conversion_script,
    generate_ollama_modelfile,
)


def test_generate_ollama_modelfile(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    tok_cfg = {
        "chat_template": "{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>{{ end }}",
        "eos_token": "<|eot_id|>",
    }
    with open(model_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f)

    modelfile = generate_ollama_modelfile(model_dir)
    assert modelfile.exists()
    content = modelfile.read_text(encoding="utf-8")

    assert "FROM ./model.safetensors" in content
    assert "PARAMETER temperature" in content
    assert '<|start_header_id|>' in content


def test_generate_gguf_conversion_script(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    script = generate_gguf_conversion_script(model_dir)
    assert script.exists()
    content = script.read_text(encoding="utf-8")
    assert "convert_hf_to_gguf.py" in content
    assert "llama-quantize" in content
    assert "Q4_K_M" in content
