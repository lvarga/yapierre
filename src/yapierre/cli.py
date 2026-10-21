"""yapierre command-line interface."""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__
from .config import CONFIG_TEMPLATE, load_config, user_config_path
from .effects import all_effects, detect_effects, process
from .engines import ENGINES, SynthesisRequest, load_engine

EXAMPLES = """\
examples:
  yapierre "Hello world"
  yapierre "Guten Tag!" -l de -o hallo.mp3
  yapierre "I am your new overlord." --style "make it sound like a robot"
  yapierre "Once upon a time..." --voice "an old man with a warm, raspy voice" --style "slow and mysterious"
  yapierre "Breaking news!" --speaker Ryan --style "excited" --effect radio
  yapierre -f story.txt --ref-audio me.wav --ref-text "Transcript of me.wav" -o story.wav
  echo "Piped text" | yapierre - --play
  yapierre --list effects
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yapierre",
        description="Turn text into speech with AI voices, style instructions and effects.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", nargs="?", help="text to speak, or '-' to read from stdin")
    p.add_argument("-f", "--file", help="read the text from a file")
    p.add_argument("-o", "--output", help="output file; format from extension (.wav .mp3 .ogg .flac)")
    p.add_argument("-l", "--lang", help="language code or name (en, de, fr, ... or auto)")
    p.add_argument("-s", "--style", help='how to say it, e.g. "whispering, scared" or "sound like a robot"')
    p.add_argument("-v", "--voice", help='describe the speaker, e.g. "a young woman with a bright voice"')
    p.add_argument("--speaker", help="use a preset speaker (see --list speakers)")
    p.add_argument("--ref-audio", help="clone the voice from this recording (3+ seconds)")
    p.add_argument("--ref-text", help="transcript of --ref-audio (improves cloning quality)")
    p.add_argument("-e", "--effect", action="append", default=[], help="add an audio effect; repeatable (see --list effects)")
    p.add_argument("--no-auto-effects", action="store_true", help="don't pick effects from words in --style")
    p.add_argument("--engine", choices=list(ENGINES), help="TTS engine")
    p.add_argument("--device", help="auto, cuda, cuda:0, cpu, mps")
    p.add_argument("--seed", type=int, help="random seed for repeatable results")
    p.add_argument("--play", action="store_true", help="play the result when done (needs ffplay)")
    p.add_argument("--config", help="extra config file to load")
    p.add_argument("--list", choices=["engines", "effects", "languages", "speakers"], help="show options and exit")
    p.add_argument("--init-config", action="store_true", help=f"write a starter config to {user_config_path()}")
    p.add_argument("--verbose", action="store_true", help="show model loading warnings and details")
    p.add_argument("--version", action="version", version=f"yapierre {__version__}")
    return p


def read_text(args) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text == "-" or (args.text is None and not sys.stdin.isatty()):
        return sys.stdin.read()
    return args.text or ""


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long text into chunks at sentence boundaries so the model stays coherent."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n\s*\n", text)
    chunks, current = [], ""
    for sentence in filter(None, (s.strip() for s in sentences)):
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def default_output(config: dict) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(config["output_dir"]) / f"yapierre_{stamp}.{config['output_format'].lstrip('.')}"


def play(path: Path) -> None:
    import shutil
    import subprocess

    ffplay = shutil.which("ffplay")
    if not ffplay:
        print("yapierre: --play needs ffplay (comes with ffmpeg); skipping playback.")
        return
    subprocess.run([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], check=False)


def list_options(what: str, config: dict, engine_factory) -> None:
    if what == "engines":
        for name, (_, desc) in ENGINES.items():
            print(f"{name:10} {desc}")
    elif what == "effects":
        for e in all_effects(config).values():
            keys = f"  (auto: {', '.join(e.keywords)})" if e.keywords else ""
            print(f"{e.name:12} {e.description}{keys}")
    elif what == "languages":
        print("\n".join(engine_factory().languages()))
    elif what == "speakers":
        print("\n".join(engine_factory().speakers()))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, loaded = load_config(args.config)
    if args.verbose and loaded:
        print("yapierre: config from " + ", ".join(map(str, loaded)))

    if args.init_config:
        path = user_config_path()
        if path.exists():
            raise SystemExit(f"yapierre: {path} already exists; edit it directly.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        print(f"Wrote {path}")
        return 0

    for key in ("engine", "device"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if config.get("cache_dir"):
        # Set before any Hugging Face import so all downloads land there.
        os.environ["HF_HOME"] = str(Path(config["cache_dir"]).expanduser())

    engine_name = config["engine"]

    def engine_factory():
        cls = load_engine(engine_name)
        return cls(config.get("engines", {}).get(engine_name, {}), config, verbose=args.verbose)

    if args.list:
        list_options(args.list, config, engine_factory)
        return 0

    text = read_text(args).strip()
    if not text:
        build_parser().print_usage()
        raise SystemExit("yapierre: no text given")

    effects = all_effects(config)
    chosen = list(args.effect)
    unknown = [e for e in chosen if e not in effects]
    if unknown:
        raise SystemExit(f"yapierre: unknown effect(s): {', '.join(unknown)}. See: yapierre --list effects")
    if config.get("auto_effects", True) and not args.no_auto_effects:
        for name in detect_effects(args.style, effects):
            if name not in chosen:
                chosen.append(name)
                print(f"yapierre: adding '{name}' effect (from --style)")

    if args.ref_text and not args.ref_audio:
        raise SystemExit("yapierre: --ref-text needs --ref-audio")
    if args.ref_audio and not Path(args.ref_audio).is_file():
        raise SystemExit(f"yapierre: reference audio not found: {args.ref_audio}")

    output = Path(args.output) if args.output else default_output(config)
    output.parent.mkdir(parents=True, exist_ok=True)

    engine = engine_factory()
    language = config["language"] if args.lang is None else args.lang
    engine.resolve_language(language)  # fail fast on unsupported languages

    if args.seed is not None:
        set_seed(args.seed)

    chunks = split_text(text, int(config.get("max_chunk_chars", 400)))
    print(f"yapierre: generating with {engine_name} ({len(text)} chars, {len(chunks)} chunk(s))...")
    started = time.time()
    pieces, sample_rate = [], None
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"  chunk {i}/{len(chunks)}")
        request = SynthesisRequest(
            text=chunk,
            language=language,
            style=args.style,
            voice=args.voice,
            speaker=args.speaker,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
        wav, sample_rate = engine.synthesize(request)
        pieces.append(wav)
        if i < len(chunks):
            pieces.append(np.zeros(int(sample_rate * 0.25), dtype=np.float32))
    audio = np.concatenate(pieces)

    needs_ffmpeg = bool(chosen) or output.suffix.lower() not in (".wav", ".flac")
    if needs_ffmpeg:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.wav"
            sf.write(raw, audio, sample_rate)
            process(raw, output, chosen, effects, sample_rate)
    else:
        sf.write(output, audio, sample_rate)

    seconds = len(audio) / sample_rate
    extra = f", effects: {', '.join(chosen)}" if chosen else ""
    print(f"yapierre: saved {output} ({seconds:.1f}s of audio in {time.time() - started:.1f}s{extra})")

    if args.play:
        play(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
