"""Text-to-speech adapter used before Talking Face generation."""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from src.config import tts as config


_STICKER_LABEL_RE = re.compile(
    r"[\[（(【]\s*(?:貼圖|表情符號|表情貼|sticker|emoji)"
    r"(?:\s*[:：-]\s*[^\]）)】\r\n]{0,40})?\s*[\]）)】]",
    re.IGNORECASE,
)
_LINE_EMOJI_ALT_RE = re.compile(r"\([A-Za-z][A-Za-z0-9 _-]{0,30}\)")


def sanitize_tts_text(text: str) -> str:
    """Remove visual-only emoji/sticker labels without changing ERC input."""
    import emoji

    cleaned = emoji.replace_emoji(text or "", replace="")
    cleaned = _STICKER_LABEL_RE.sub("", cleaned)
    # Older/default LINE emoji may arrive only as undisclosed alternative text
    # such as ``(love)`` and without an ``emojis`` metadata entry.
    cleaned = _LINE_EMOJI_ALT_RE.sub("", cleaned)
    cleaned = cleaned.replace("\ufffc", "").replace("\u200d", "").replace("\ufe0f", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([，。！？,.!?])", r"\1", cleaned)
    return cleaned


class TextToSpeechError(RuntimeError):
    """Raised when speech audio cannot be produced."""


@dataclass(frozen=True)
class TextToSpeechResult:
    output_path: Path
    processing_ms: int
    voice: str
    emotion: str = "neutral"
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"


class TextToSpeechAdapter(Protocol):
    async def synthesize(
        self, text: str, output_path: str | Path, *, emotion: str = "neutral"
    ) -> TextToSpeechResult: ...


CommunicateFactory = Callable[..., Any]


def _edge_communicate_factory(text: str, voice: str, **kwargs):
    import edge_tts

    return edge_tts.Communicate(text, voice, **kwargs)


class EdgeTextToSpeechAdapter:
    """Generate MP3 speech with Microsoft Edge's online TTS service."""

    EMOTION_PROFILES = {
        "neutral": {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
        "joy": {"rate": "+12%", "volume": "+5%", "pitch": "+12Hz"},
        "sadness": {"rate": "-18%", "volume": "-8%", "pitch": "-12Hz"},
        "anger": {"rate": "+8%", "volume": "+12%", "pitch": "-4Hz"},
        "surprise": {"rate": "+15%", "volume": "+5%", "pitch": "+18Hz"},
        "fear": {"rate": "+6%", "volume": "-3%", "pitch": "+14Hz"},
        "disgust": {"rate": "-10%", "volume": "-3%", "pitch": "-8Hz"},
    }

    def __init__(
        self,
        *,
        voice: str = "zh-TW-YunJheNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        timeout: float = 60.0,
        max_text_chars: int = 500,
        communicate_factory: CommunicateFactory = _edge_communicate_factory,
    ) -> None:
        if not voice.strip():
            raise ValueError("TTS voice must not be empty")
        if timeout <= 0:
            raise ValueError("TTS timeout must be positive")
        if max_text_chars <= 0:
            raise ValueError("TTS text limit must be positive")
        self.voice = voice.strip()
        self.rate = rate.strip()
        self.volume = volume.strip()
        self.pitch = pitch.strip()
        self.timeout = float(timeout)
        self.max_text_chars = int(max_text_chars)
        self._communicate_factory = communicate_factory

    async def synthesize(
        self, text: str, output_path: str | Path, *, emotion: str = "neutral"
    ) -> TextToSpeechResult:
        clean_text = sanitize_tts_text(text)
        if not clean_text:
            raise TextToSpeechError("TTS text has no speakable content")
        if not any(character.isalnum() for character in clean_text):
            raise TextToSpeechError("TTS text has no speakable content")
        if len(clean_text) > self.max_text_chars:
            raise TextToSpeechError(
                f"TTS text exceeds {self.max_text_chars} characters"
            )
        output = Path(output_path).resolve()
        if output.suffix.lower() != ".mp3":
            raise TextToSpeechError("TTS output path must end in .mp3")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.part")
        normalised_emotion = (emotion or "neutral").strip().lower()
        profile = self.EMOTION_PROFILES.get(normalised_emotion)
        if profile is None:
            normalised_emotion = "neutral"
            profile = {
                "rate": self.rate,
                "volume": self.volume,
                "pitch": self.pitch,
            }
        elif normalised_emotion == "neutral":
            profile = {
                "rate": self.rate,
                "volume": self.volume,
                "pitch": self.pitch,
            }
        started = time.perf_counter()
        try:
            communicator = self._communicate_factory(
                clean_text,
                self.voice,
                rate=profile["rate"],
                volume=profile["volume"],
                pitch=profile["pitch"],
            )
            await asyncio.wait_for(
                communicator.save(str(temporary)), timeout=self.timeout
            )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise TextToSpeechError("TTS returned an empty audio file")
            await asyncio.to_thread(os.replace, temporary, output)
        except TextToSpeechError:
            temporary.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            temporary.unlink(missing_ok=True)
            raise TextToSpeechError(
                f"TTS generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        return TextToSpeechResult(
            output_path=output,
            processing_ms=round((time.perf_counter() - started) * 1000),
            voice=self.voice,
            emotion=normalised_emotion,
            rate=profile["rate"],
            volume=profile["volume"],
            pitch=profile["pitch"],
        )


def create_tts_adapter() -> EdgeTextToSpeechAdapter:
    return EdgeTextToSpeechAdapter(
        voice=config.TTS_VOICE,
        rate=config.TTS_RATE,
        volume=config.TTS_VOLUME,
        pitch=config.TTS_PITCH,
        timeout=config.TTS_TIMEOUT,
        max_text_chars=config.TTS_MAX_TEXT_CHARS,
    )
