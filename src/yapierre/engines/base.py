"""The interface every TTS engine implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SynthesisRequest:
    text: str
    # Normalized language name ("english") or "auto".
    language: str = "auto"
    # How to say it: "angry and fast", "make it sound like a robot".
    style: str | None = None
    # Who says it, described in words: "an old man with a raspy voice".
    voice: str | None = None
    # A preset speaker name, for engines that have them.
    speaker: str | None = None
    # Reference recording to clone, plus its transcript.
    ref_audio: str | None = None
    ref_text: str | None = None


class Engine(ABC):
    """A text-to-speech backend.

    `settings` is the engine's own config table ([engines.<name>]); `common` holds the
    top-level settings shared by all engines (device, cache_dir, ...).
    """

    name: str = ""
    description: str = ""

    def __init__(self, settings: dict, common: dict, verbose: bool = False):
        self.settings = settings
        self.common = common
        self.verbose = verbose

    @abstractmethod
    def synthesize(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        """Return (mono float32 waveform, sample rate)."""

    def languages(self) -> list[str]:
        return []

    def speakers(self) -> list[str]:
        return []

    def resolve_language(self, language: str) -> str:
        return language
