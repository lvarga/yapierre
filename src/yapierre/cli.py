"""yapierre command-line interface."""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__
from .config import CONFIG_TEMPLATE, load_config, user_config_path
from .effects import all_effects, detect_effects, process
from .engines import ENGINES, Engine, SynthesisRequest, load_engine

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

many jobs, models loaded once (each line takes the same options as above):
  yapierre --batch jobs.txt              # run every line of a file
  yapierre --batch jobs.txt -l de -e radio   # options here are defaults for every line
  yapierre --interactive                 # type jobs one at a time
"""

# Options that configure the whole run and can't change from one batch line to the next.
SESSION_OPTIONS = ("batch", "interactive", "init_config", "config", "device", "verbose")


class _JobParser(argparse.ArgumentParser):
    """Parser for batch/interactive lines: report errors without printing the full usage."""

    def error(self, message):
        raise SystemExit(f"yapierre: {message}")


def build_parser(cls: type[argparse.ArgumentParser] = argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = cls(
        prog="yapierre",
        description="Turn text into speech with AI voices, style instructions and effects.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("text", nargs="*", help="text to speak, or '-' to read from stdin")
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
    p.add_argument("--batch", metavar="FILE", help="run one job per line of FILE ('-' for stdin), loading models once")
    p.add_argument("-i", "--interactive", action="store_true", help="read jobs from the prompt, keeping models loaded")
    p.add_argument("--config", help="extra config file to load")
    p.add_argument("--list", choices=["engines", "effects", "languages", "speakers"], help="show options and exit")
    p.add_argument("--init-config", action="store_true", help=f"write a starter config to {user_config_path()}")
    p.add_argument("--verbose", action="store_true", help="show model loading warnings and details")
    p.add_argument("--version", action="version", version=f"yapierre {__version__}")
    return p


def read_text(args, allow_stdin: bool) -> str:
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8")
        except OSError as e:
            raise SystemExit(f"yapierre: can't read {args.file}: {e.strerror}")
    wants_stdin = args.text == ["-"] or (not args.text and allow_stdin and not sys.stdin.isatty())
    if wants_stdin:
        if not allow_stdin:
            raise SystemExit("yapierre: '-' (read from stdin) can't be used in batch or interactive mode")
        return sys.stdin.read()
    return " ".join(args.text)


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


def play(path: Path) -> None:
    import shutil
    import subprocess

    ffplay = shutil.which("ffplay")
    if not ffplay:
        print("yapierre: --play needs ffplay (comes with ffmpeg); skipping playback.")
        return
    subprocess.run([ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)], check=False)


def report_exit(e: SystemExit) -> None:
    """Print the message of a SystemExit caught inside batch/interactive mode."""
    if isinstance(e.code, str):
        print(e.code, file=sys.stderr)


@dataclass
class Job:
    request: SynthesisRequest
    engine: str
    effects: list[str]
    output: Path
    seed: int | None = None
    play: bool = False
    # Where the job came from, e.g. "line 3", for progress messages.
    label: str = ""


class Session:
    """Keeps engines (and their loaded models) alive so many jobs can run without reloading."""

    def __init__(self, config: dict, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.effects = all_effects(config)
        self._engines: dict[str, Engine] = {}
        self._reserved: set[Path] = set()

    def engine(self, name: str) -> Engine:
        if name not in self._engines:
            cls = load_engine(name)
            settings = self.config.get("engines", {}).get(name, {})
            self._engines[name] = cls(settings, self.config, verbose=self.verbose)
        return self._engines[name]

    def _default_output(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(self.config["output_dir"]) / f"yapierre_{stamp}.{self.config['output_format'].lstrip('.')}"
        path, n = base, 2
        while path.exists() or path in self._reserved:
            path = base.with_name(f"{base.stem}_{n}{base.suffix}")
            n += 1
        self._reserved.add(path)
        return path

    def prepare(self, args, allow_stdin: bool = True, label: str = "") -> Job:
        """Validate one job's options and turn them into a Job. Raises SystemExit on bad input."""
        text = read_text(args, allow_stdin).strip()
        if not text:
            raise SystemExit("yapierre: no text given")

        chosen = list(args.effect)
        unknown = [e for e in chosen if e not in self.effects]
        if unknown:
            raise SystemExit(f"yapierre: unknown effect(s): {', '.join(unknown)}. See: yapierre --list effects")
        if self.config.get("auto_effects", True) and not args.no_auto_effects:
            for name in detect_effects(args.style, self.effects):
                if name not in chosen:
                    chosen.append(name)
                    where = f"{label}: " if label else ""
                    print(f"yapierre: {where}adding '{name}' effect (from --style)")

        if args.ref_text and not args.ref_audio:
            raise SystemExit("yapierre: --ref-text needs --ref-audio")
        if args.ref_audio and not Path(args.ref_audio).is_file():
            raise SystemExit(f"yapierre: reference audio not found: {args.ref_audio}")

        engine_name = args.engine or self.config["engine"]
        language = self.config["language"] if args.lang is None else args.lang
        self.engine(engine_name).resolve_language(language)  # fail fast on unsupported languages

        request = SynthesisRequest(
            text=text,
            language=language,
            style=args.style,
            voice=args.voice,
            speaker=args.speaker,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
        )
        output = Path(args.output) if args.output else self._default_output()
        return Job(request, engine_name, chosen, output, args.seed, args.play, label)

    def run(self, job: Job) -> Path:
        engine = self.engine(job.engine)
        job.output.parent.mkdir(parents=True, exist_ok=True)
        if job.seed is not None:
            set_seed(job.seed)

        chunks = split_text(job.request.text, int(self.config.get("max_chunk_chars", 400)))
        print(f"yapierre: generating with {job.engine} ({len(job.request.text)} chars, {len(chunks)} chunk(s))...")
        started = time.time()
        pieces, sample_rate = [], None
        for i, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                print(f"  chunk {i}/{len(chunks)}")
            wav, sample_rate = engine.synthesize(replace(job.request, text=chunk))
            pieces.append(wav)
            if i < len(chunks):
                pieces.append(np.zeros(int(sample_rate * 0.25), dtype=np.float32))
        audio = np.concatenate(pieces)

        if job.effects or job.output.suffix.lower() not in (".wav", ".flac"):
            with tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp) / "raw.wav"
                sf.write(raw, audio, sample_rate)
                process(raw, job.output, job.effects, self.effects, sample_rate)
        else:
            sf.write(job.output, audio, sample_rate)

        seconds = len(audio) / sample_rate
        extra = f", effects: {', '.join(job.effects)}" if job.effects else ""
        print(f"yapierre: saved {job.output} ({seconds:.1f}s of audio in {time.time() - started:.1f}s{extra})")
        if job.play:
            play(job.output)
        return job.output


def list_options(what: str, session: Session, engine_name: str) -> None:
    if what == "engines":
        for name, (_, desc) in ENGINES.items():
            print(f"{name:10} {desc}")
    elif what == "effects":
        for e in session.effects.values():
            keys = f"  (auto: {', '.join(e.keywords)})" if e.keywords else ""
            print(f"{e.name:12} {e.description}{keys}")
    elif what == "languages":
        print("\n".join(session.engine(engine_name).languages()))
    elif what == "speakers":
        print("\n".join(session.engine(engine_name).speakers()))


def parse_line(parser: argparse.ArgumentParser, line: str, defaults: argparse.Namespace) -> argparse.Namespace:
    """Parse one batch/interactive line, starting from the options given when yapierre was launched."""
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.escape = ""  # keep Windows paths like C:\voices\me.wav intact
    try:
        tokens = list(lexer)
    except ValueError as e:
        raise SystemExit(f"yapierre: {e}")
    args = parser.parse_args(tokens, namespace=copy.deepcopy(defaults))
    for dest in SESSION_OPTIONS:
        if getattr(args, dest) != getattr(defaults, dest):
            raise SystemExit(f"yapierre: --{dest.replace('_', '-')} can only be used when starting yapierre")
    return args


def run_batch(args: argparse.Namespace, session: Session) -> int:
    source = args.batch
    try:
        content = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"yapierre: can't read batch file {source}: {e.strerror}")

    # Validate every line before generating anything, so a typo on line 40 doesn't waste a long run.
    parser = build_parser(_JobParser)
    jobs, bad = [], 0
    for lineno, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            job_args = parse_line(parser, line, args)
            if job_args.list:
                raise SystemExit("yapierre: --list can't be used in a batch file")
            jobs.append(session.prepare(job_args, allow_stdin=False, label=f"line {lineno}"))
        except SystemExit as e:
            print(f"line {lineno}: ", end="", file=sys.stderr)
            report_exit(e)
            bad += 1
    if bad:
        raise SystemExit(f"yapierre: {bad} invalid line(s) in {source}; nothing was generated.")
    if not jobs:
        raise SystemExit(f"yapierre: no jobs in {source}")

    # Run jobs that need the same model back to back, so each model loads only once.
    jobs.sort(key=lambda j: (j.engine, session.engine(j.engine).model_key(j.request)))
    started, failed = time.time(), []
    for i, job in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {job.label}: {job.request.text[:60]!r}")
        try:
            session.run(job)
        except SystemExit as e:
            report_exit(e)
            failed.append(job.label)
        except KeyboardInterrupt:
            print("\nyapierre: batch cancelled")
            return 130
        except Exception as e:  # noqa: BLE001 - keep going; one bad job shouldn't lose the rest
            print(f"yapierre: {job.label} failed: {e}", file=sys.stderr)
            failed.append(job.label)

    done = len(jobs) - len(failed)
    print(f"\nyapierre: batch finished: {done}/{len(jobs)} succeeded in {time.time() - started:.0f}s")
    if failed:
        print(f"yapierre: failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


INTERACTIVE_HELP = """\
Type a job the same way as on the command line, without the "yapierre" part, e.g.
  "Hello there" -s cheerful -o hi.wav
  Guten Morgen -l de --speaker Ryan
Options given when starting yapierre apply to every job. Models stay loaded between jobs.
Commands: help (all options), quit"""


def run_interactive(args: argparse.Namespace, session: Session) -> int:
    parser = build_parser(_JobParser)
    prompt = "yapierre> " if sys.stdin.isatty() else ""
    if prompt:
        print(INTERACTIVE_HELP)
    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line or line.startswith("#"):
            continue
        if line in ("quit", "exit", "q"):
            return 0
        if line in ("help", "?"):
            parser.print_help()
            continue
        try:
            job_args = parse_line(parser, line, args)
            if job_args.list:
                list_options(job_args.list, session, job_args.engine or session.config["engine"])
                continue
            session.run(session.prepare(job_args, allow_stdin=False))
        except SystemExit as e:
            report_exit(e)
        except KeyboardInterrupt:
            print("\nyapierre: cancelled")
        except Exception as e:  # noqa: BLE001 - keep the session (and its loaded models) alive
            print(f"yapierre: error: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    # Echoing text the console encoding can't show (e.g. Chinese on a cp1252 pipe) must not crash a job.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
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

    if args.device:
        config["device"] = args.device
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if config.get("cache_dir"):
        # Set before any Hugging Face import so all downloads land there.
        os.environ["HF_HOME"] = str(Path(config["cache_dir"]).expanduser())

    session = Session(config, verbose=args.verbose)

    if args.list:
        list_options(args.list, session, args.engine or config["engine"])
        return 0

    if args.batch or args.interactive:
        if args.batch and args.interactive:
            raise SystemExit("yapierre: use either --batch or --interactive, not both")
        if args.text or args.file or args.output:
            raise SystemExit("yapierre: give text and -o per line in batch/interactive mode, not on the command line")
        return run_batch(args, session) if args.batch else run_interactive(args, session)

    session.run(session.prepare(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
