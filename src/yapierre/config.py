"""Configuration loading.

Settings are merged in this order (later wins):
built-in defaults < user config < project config (./yapierre.toml) < --config file < CLI flags.
"""

from __future__ import annotations

import copy
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "engine": "qwen",
    "language": "auto",
    "output_dir": ".",
    "output_format": "wav",
    # Where AI models are downloaded. Empty = Hugging Face default (~/.cache/huggingface).
    "cache_dir": "",
    # auto | cuda | cuda:0 | cpu | mps
    "device": "auto",
    # Apply effects automatically when the style mentions them ("sound like a robot" -> robot).
    "auto_effects": True,
    # Long texts are split into chunks of roughly this many characters.
    "max_chunk_chars": 400,
    "engines": {
        "qwen": {
            "dtype": "bfloat16",
            # Models kept in memory at once in batch/interactive mode (~4.5 GB VRAM each).
            "max_loaded_models": 1,
            # Used when neither --voice, --speaker nor --ref-audio is given.
            "default_voice": "A clear, natural, friendly adult voice.",
            "models": {
                "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            },
            "generation": {},
        },
    },
    # User-defined effects: [effects.name] filter = "<ffmpeg -af chain>", keywords = [...]
    "effects": {},
}

CONFIG_TEMPLATE = """\
# yapierre configuration
# Lookup order (later wins): user config < ./yapierre.toml < --config FILE < command-line flags.

engine = "qwen"
language = "auto"          # auto, or a code/name: en, de, fr, es, it, pt, ru, zh, ja, ko
output_dir = "."
output_format = "wav"      # wav, mp3, ogg, flac (anything ffmpeg can write)
cache_dir = ""             # where models are downloaded; empty = Hugging Face default
device = "auto"            # auto, cuda, cuda:0, cpu, mps
auto_effects = true        # "sound like a robot" in --style also applies the robot effect
max_chunk_chars = 400

[engines.qwen]
dtype = "bfloat16"         # bfloat16, float16, float32 (use float32 on CPU)
max_loaded_models = 1      # models kept in memory in batch/interactive mode (~4.5 GB VRAM each)
default_voice = "A clear, natural, friendly adult voice."

[engines.qwen.models]
# Smaller, faster alternatives: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, Qwen/Qwen3-TTS-12Hz-0.6B-Base
design = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
custom = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
clone = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

[engines.qwen.generation]
# Passed straight to the model, e.g.:
# temperature = 0.9
# top_p = 1.0
# max_new_tokens = 2048

# Add your own ffmpeg effects (see `yapierre --list effects` for built-ins):
# [effects.stadium]
# filter = "aecho=0.8:0.85:250|500:0.4|0.3,volume=1.2"
# keywords = ["stadium", "arena"]
"""


def user_config_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "yapierre" / "config.toml"


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"yapierre: invalid config file {path}: {e}")


def load_config(explicit: str | None = None) -> tuple[dict, list[Path]]:
    """Return the merged config and the list of files that were loaded."""
    config = copy.deepcopy(DEFAULTS)
    candidates = [user_config_path(), Path.cwd() / "yapierre.toml"]
    if env := os.environ.get("YAPIERRE_CONFIG"):
        candidates.append(Path(env))
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"yapierre: config file not found: {path}")
        candidates.append(path)

    loaded = []
    for path in candidates:
        if path.is_file() and path.resolve() not in [p.resolve() for p in loaded]:
            _merge(config, _read(path))
            loaded.append(path)
    return config, loaded
