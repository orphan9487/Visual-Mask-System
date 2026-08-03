from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.generation.renderers.base import GenerationRequest
from src.generation.renderers import static_sticker_renderer as renderer_module
from src.generation.renderers.static_sticker_renderer import StaticStickerRenderer


class _FakeGenerator:
    def __init__(self, *, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakePostQuantConv:
    @staticmethod
    def parameters():
        return iter([SimpleNamespace(dtype=torch.float32)])


class _FakeVAE:
    post_quant_conv = _FakePostQuantConv()

    @staticmethod
    def decode(latents, **kwargs):
        return latents


class _FakePipeline:
    def __init__(self, outputs):
        self.vae = _FakeVAE()
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs)


def _request(seed=100):
    return GenerationRequest(
        identity={},
        prompt="portrait",
        negative_prompt="low quality",
        num_frames=1,
        num_inference_steps=2,
        guidance_scale=7.5,
        width=64,
        height=64,
        seed=seed,
    )


def _result(color, *, rejected=False):
    return SimpleNamespace(
        images=[Image.new("RGB", (8, 8), color)],
        nsfw_content_detected=[rejected],
    )


def _renderer(outputs):
    renderer = StaticStickerRenderer.__new__(StaticStickerRenderer)
    renderer.pipe = _FakePipeline(outputs)
    renderer._load_lora = lambda identity: None
    return renderer


def test_retries_safety_rejected_black_image(monkeypatch):
    renderer = _renderer([
        _result("black", rejected=True),
        _result("white"),
    ])
    monkeypatch.setattr(renderer_module.torch, "Generator", _FakeGenerator)
    monkeypatch.setattr(renderer_module.random, "randint", lambda start, end: 101)

    rendered = renderer.render(_request())

    assert rendered.seed == 101
    assert rendered.frames[0].getpixel((0, 0)) == (255, 255, 255)
    assert len(renderer.pipe.calls) == 2


def test_raises_when_every_attempt_is_black(monkeypatch):
    renderer = _renderer([_result("black") for _ in range(3)])
    seeds = iter([101, 102])
    monkeypatch.setattr(renderer_module.torch, "Generator", _FakeGenerator)
    monkeypatch.setattr(renderer_module.random, "randint", lambda start, end: next(seeds))

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        renderer.render(_request())

    assert len(renderer.pipe.calls) == 3
