"""Audio effects applied with ffmpeg after synthesis."""

from __future__ import annotations

import json
import math
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


def _measure_loudness(ffmpeg: str, src: Path, filters: list[str], loudness: float, true_peak: float) -> dict | None:
    """First loudnorm pass: measure the clip (after effects). None if it is silent."""
    probe = f"loudnorm=I={loudness}:TP={true_peak}:LRA=11:print_format=json"
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(src), "-af", ",".join([*filters, probe]), "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"yapierre: ffmpeg failed while measuring loudness:\n{result.stderr.strip()}")
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr)
    if not match:
        raise SystemExit("yapierre: couldn't read ffmpeg's loudness measurement")
    stats = json.loads(match.group(0))
    if not math.isfinite(float(stats["input_i"])):
        return None
    return stats


def process(
    src: Path,
    dst: Path,
    effect_names: list[str],
    effects: dict[str, Effect],
    sample_rate: int,
    loudness: float | None = None,
    true_peak: float = -1.5,
) -> None:
    """Run src through the effect chain with ffmpeg and write dst (format from its extension).

    With `loudness` (LUFS), the result is normalized to that integrated loudness (EBU R128): a first
    pass measures it, the second applies the exact gain needed plus a peak limiter at `true_peak`.
    (loudnorm's own linear mode undershoots on speech, whose sharp peaks would cross the ceiling.)
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit(
            "yapierre: ffmpeg is required for effects, loudness normalization and non-WAV output,\n"
            "but was not found on PATH. Install it from https://ffmpeg.org/download.html"
        )
    filters = [effects[n].filter.replace("{sr}", str(sample_rate)) for n in effect_names]
    if filters:
        # Stacked effects can push the level past 0 dBFS; a limiter keeps it from clipping.
        filters.append("alimiter=limit=0.95:level=0")
    stats = _measure_loudness(ffmpeg, src, filters, loudness, true_peak) if loudness is not None else None
    if not stats:
        _render(ffmpeg, src, dst, filters)
        return

    gain = loudness - float(stats["input_i"])
    # alimiter limits sample peaks; leave 0.5 dB of headroom for inter-sample (true) peaks.
    ceiling = min(1.0, max(0.0625, 10 ** ((true_peak - 0.5) / 20)))
    limiter = f"alimiter=limit={ceiling:.4f}:attack=2:release=60:level=0"
    # Limiting peaky audio (e.g. the robot effect) lowers its loudness a little, so measure the
    # result and correct the gain; it converges within a pass or two.
    for _ in range(3):
        gain = min(gain, 40.0)  # don't blow up near-silent clips
        _render(ffmpeg, src, dst, [*filters, f"volume={gain:.2f}dB", limiter])
        result = _measure_loudness(ffmpeg, dst, [], loudness, true_peak)
        if not result:
            break
        error = loudness - float(result["input_i"])
        if abs(error) <= 0.3:
            break
        gain += error


def _render(ffmpeg: str, src: Path, dst: Path, filters: list[str]) -> None:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd.append(str(dst))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"yapierre: ffmpeg failed:\n{result.stderr.strip()}")
