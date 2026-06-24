"""Tests for transcription backend helpers."""

from __future__ import annotations

import sys
import types

from knowledge_distiller import transcriber


class _FakeParameter:
    def __init__(self, device: str):
        self.device = device


class _FakeTorchModel:
    dtype = "float16"

    def __init__(self):
        self.device = "cpu"
        self.eval_called = False

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        yield _FakeParameter(self.device)


class _FakeQwen3ASRModel:
    last_kwargs: dict | None = None

    def __init__(self):
        self.model = _FakeTorchModel()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.last_kwargs = {"model_id": model_id, **kwargs}
        return cls()


def _install_fake_torch(monkeypatch, *, cuda: bool, mps: bool):
    fake_torch = types.SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def _install_fake_qwen_asr(monkeypatch):
    _FakeQwen3ASRModel.last_kwargs = None
    fake_qwen_asr = types.SimpleNamespace(Qwen3ASRModel=_FakeQwen3ASRModel)
    monkeypatch.setitem(sys.modules, "qwen_asr", fake_qwen_asr)


def test_select_torch_device_prefers_cuda(monkeypatch):
    _install_fake_torch(monkeypatch, cuda=True, mps=True)

    assert transcriber._select_torch_device() == "cuda"


def test_select_torch_device_falls_back_to_mps(monkeypatch):
    _install_fake_torch(monkeypatch, cuda=False, mps=True)

    assert transcriber._select_torch_device() == "mps"


def test_select_torch_device_falls_back_to_cpu(monkeypatch):
    _install_fake_torch(monkeypatch, cuda=False, mps=False)

    assert transcriber._select_torch_device() == "cpu"


def test_load_qwen3_asr_model_moves_model_to_best_device(monkeypatch):
    _install_fake_torch(monkeypatch, cuda=False, mps=True)
    _install_fake_qwen_asr(monkeypatch)

    model = transcriber._load_qwen3_asr_model("Qwen/Qwen3-ASR-0.6B", "0.6b")

    assert _FakeQwen3ASRModel.last_kwargs == {
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "torch_dtype": "float16",
    }
    assert model.model.device == "mps"
    assert model.device == "mps"
    assert model.dtype == "float16"
    assert model.model.eval_called is True


def test_load_qwen3_asr_model_uses_float32_on_cpu(monkeypatch):
    _install_fake_torch(monkeypatch, cuda=False, mps=False)
    _install_fake_qwen_asr(monkeypatch)

    transcriber._load_qwen3_asr_model("Qwen/Qwen3-ASR-0.6B", "0.6b")

    assert _FakeQwen3ASRModel.last_kwargs == {
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "torch_dtype": "float32",
    }
