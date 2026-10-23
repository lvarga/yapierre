"""Qwen3-TTS engine (https://github.com/QwenLM/Qwen3-TTS).

Picks one of three models per request:
- clone:  --ref-audio given           -> Base model, copies the reference voice
- custom: --speaker given             -> CustomVoice model, preset speaker + style instruction
- design: otherwise                   -> VoiceDesign model, voice described in words + style
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import warnings

import numpy as np

from .base import Engine, SynthesisRequest

LANGUAGE_CODES = {
    "en": "english",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}


@contextlib.contextmanager
def _quiet(enabled: bool):
    """Silence the warnings qwen-tts and its dependencies print on import and load."""
    if not enabled:
        yield
        return
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    # Some dependencies spawn shell commands (e.g. a `sox` probe) that write straight to the
    # console, so redirect the stderr file descriptor itself, not just sys.stderr.
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull)
        logging.disable(previous)


class QwenEngine(Engine):
    name = "qwen"
    description = "Qwen3-TTS"

    def __init__(self, settings: dict, common: dict, verbose: bool = False):
        super().__init__(settings, common, verbose)
        self._models: dict[str, object] = {}
        self._speakers: list[str] | None = None

    # --- model loading -------------------------------------------------------------

    def _device_and_dtype(self):
        import torch

        device = self.common.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype_name = self.settings.get("dtype", "bfloat16")
        if device == "cpu" and dtype_name != "float32":
            dtype_name = "float32"  # half precision on CPU is slow or unsupported
        return device, getattr(torch, dtype_name)

    def _unload_oldest(self) -> None:
        import gc

        import torch

        kind = next(iter(self._models))
        del self._models[kind]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.verbose:
            print(f"yapierre: unloaded {self.settings['models'][kind]} to free memory")

    def _model(self, kind: str):
        if kind in self._models:
            self._models[kind] = self._models.pop(kind)  # mark as most recently used
            return self._models[kind]
        # Each 1.7B model needs ~4.5 GB of VRAM; by default keep only one loaded at a time.
        while self._models and len(self._models) >= max(1, int(self.settings.get("max_loaded_models", 1))):
            self._unload_oldest()
        repo = self.settings["models"][kind]
        if not os.path.isdir(repo):
            from huggingface_hub import snapshot_download, try_to_load_from_cache

            if not isinstance(try_to_load_from_cache(repo, "config.json"), str):
                print(f"yapierre: downloading {repo} (one-time, several GB)...")
                snapshot_download(repo)
        print(f"yapierre: loading {repo}...")
        with _quiet(not self.verbose):
            from qwen_tts import Qwen3TTSModel

            device, dtype = self._device_and_dtype()
            model = Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=dtype)
        self._models[kind] = model
        return model

    def _config_json(self, kind: str) -> dict:
        """Read a model's config.json without loading the weights."""
        from huggingface_hub import hf_hub_download

        repo = self.settings["models"][kind]
        if os.path.isdir(repo):
            path = os.path.join(repo, "config.json")
        else:
            path = hf_hub_download(repo, "config.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # --- Engine API ----------------------------------------------------------------

    def loaded_models(self) -> list[str]:
        return [self.settings["models"][kind] for kind in self._models]

    def model_key(self, request: SynthesisRequest) -> str:
        if request.ref_audio:
            return "clone"
        if request.speaker:
            return "custom"
        return "design"

    def languages(self) -> list[str]:
        ids = self._config_json("design")["talker_config"]["codec_language_id"]
        return ["auto"] + sorted(k for k in ids if "dialect" not in k)

    def speakers(self) -> list[str]:
        if self._speakers is None:
            self._speakers = sorted(self._config_json("custom")["talker_config"]["spk_id"])
        return self._speakers

    def resolve_language(self, language: str) -> str:
        lang = (language or "auto").strip().lower()
        lang = LANGUAGE_CODES.get(lang, LANGUAGE_CODES.get(lang.split("-")[0], lang))
        supported = set(LANGUAGE_CODES.values()) | {"auto"}
        if lang not in supported:
            raise SystemExit(
                f"yapierre: Qwen3-TTS does not support language '{language}'.\n"
                f"Supported: auto, {', '.join(f'{c} ({n})' for c, n in LANGUAGE_CODES.items())}"
            )
        return lang

    def synthesize(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        gen = dict(self.settings.get("generation") or {})
        language = self.resolve_language(request.language)
        language = "Auto" if language == "auto" else language.capitalize()

        kind = self.model_key(request)
        if kind == "clone":
            if request.style or request.voice:
                print("yapierre: note: --style/--voice are ignored when cloning a voice (--ref-audio).")
            model = self._model("clone")
            clone_args = {"ref_audio": request.ref_audio}
            if request.ref_text:
                clone_args["ref_text"] = request.ref_text
            else:
                clone_args["x_vector_only_mode"] = True
            wavs, sr = model.generate_voice_clone(text=request.text, language=language, **clone_args, **gen)
        elif kind == "custom":
            model = self._model("custom")
            if request.voice:
                print("yapierre: note: --voice is ignored when a preset --speaker is used.")
            wavs, sr = model.generate_custom_voice(
                text=request.text,
                language=language,
                speaker=request.speaker,
                instruct=request.style or "",
                **gen,
            )
        else:
            model = self._model("design")
            voice = request.voice or self.settings.get("default_voice", "")
            instruct = " ".join(p.strip().rstrip(".") + "." for p in (voice, request.style) if p and p.strip())
            wavs, sr = model.generate_voice_design(text=request.text, language=language, instruct=instruct, **gen)

        return np.asarray(wavs[0], dtype=np.float32), int(sr)
