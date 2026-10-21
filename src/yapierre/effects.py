"""Audio effects applied with ffmpeg after synthesis."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Effect:
    name: str
    description: str
    # ffmpeg -af filter chain. "{sr}" is replaced with the input sample rate.
    filter: str
    # Words in --style that trigger this effect when auto_effects is on.
    keywords: list[str] = field(default_factory=list)


BUILTIN_EFFECTS = [
    Effect(
        "robot",
        "Flat, metallic robot voice",
        "afftfilt=real='hypot(re,im)*sin(0)':imag='hypot(re,im)*cos(0)':win_size=512:overlap=0.75,"
        "aecho=0.8:0.6:8:0.4,volume=15dB",
        ["robot", "robotic", "android", "cyborg", "machine voice"],
    ),
    Effect(
        "whisper",
        "Breathy, unvoiced whisper",
        "afftfilt=real='hypot(re,im)*cos((random(0)*2-1)*2*3.14)':"
        "imag='hypot(re,im)*sin((random(1)*2-1)*2*3.14)':win_size=128:overlap=0.8,volume=6dB",
    ),
    Effect(
        "radio",
        "Old AM radio",
        "highpass=f=300,lowpass=f=3000,acompressor=threshold=0.1:ratio=8,volume=2",
        ["radio", "walkie-talkie", "walkie talkie", "broadcast"],
    ),
    Effect(
        "telephone",
        "Narrow-band phone call",
        "aresample=8000,highpass=f=300,lowpass=f=3400,aresample={sr}",
        ["telephone", "phone call", "over the phone"],
    ),
    Effect(
        "echo",
        "Short echo",
        "aecho=0.8:0.8:120:0.4",
        ["echo", "echoing"],
    ),
    Effect(
        "cave",
        "Big reverberant space",
        "aecho=0.8:0.9:400|800|1200:0.4|0.25|0.15",
        ["cave", "cathedral", "huge hall", "dungeon"],
    ),
    Effect(
        "chipmunk",
        "High-pitched cartoon voice",
        "asetrate={sr}*1.5,aresample={sr},atempo=0.6667",
        ["chipmunk", "helium", "squeaky"],
    ),
    Effect(
        "giant",
        "Very deep, monstrous voice",
        "asetrate={sr}*0.7,aresample={sr},atempo=1.4286,bass=g=4",
        ["giant", "demon", "monster", "ogre"],
    ),
    Effect(
        "alien",
        "Warbling sci-fi alien",
        "vibrato=f=6:d=0.6,flanger=delay=5:depth=4:speed=0.5",
        ["alien", "extraterrestrial"],
    ),
    Effect(
        "underwater",
        "Muffled and wobbly",
        "lowpass=f=600,vibrato=f=2:d=0.3,volume=1.5",
        ["underwater", "under water"],
    ),
]


def all_effects(config: dict) -> dict[str, Effect]:
    effects = {e.name: e for e in BUILTIN_EFFECTS}
    for name, spec in (config.get("effects") or {}).items():
        if "filter" not in spec:
            raise SystemExit(f"yapierre: effect '{name}' in config has no 'filter'")
        effects[name] = Effect(
            name,
            spec.get("description", "Custom effect"),
            spec["filter"],
            list(spec.get("keywords", [])),
        )
    return effects


def detect_effects(style: str | None, effects: dict[str, Effect]) -> list[str]:
    """Find effects whose keywords appear in the style text."""
    if not style:
        return []
    text = style.lower()
    found = []
    for effect in effects.values():
        if any(re.search(rf"\b{re.escape(k)}\b", text) for k in effect.keywords):
            found.append(effect.name)
    return found


def find_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def process(src: Path, dst: Path, effect_names: list[str], effects: dict[str, Effect], sample_rate: int) -> None:
    """Run src through the effect chain with ffmpeg and write dst (format from its extension)."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit(
            "yapierre: ffmpeg is required for effects and non-WAV output, but was not found on PATH.\n"
            "Install it from https://ffmpeg.org/download.html"
        )
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if effect_names:
        chain = ",".join(effects[n].filter.replace("{sr}", str(sample_rate)) for n in effect_names)
        # Stacked effects can push the level past 0 dBFS; a limiter keeps it from clipping.
        cmd += ["-af", f"{chain},alimiter=limit=0.95"]
    cmd.append(str(dst))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"yapierre: ffmpeg failed:\n{result.stderr.strip()}")
