"""Local web interface: `yapierre --serve`.

A small standard-library HTTP server around a cli.Session, so models stay loaded between requests.
Generation jobs run one at a time on a worker thread (the GPU can only do one at once); the page
polls for their progress. Effect previews only need ffmpeg, so they run immediately.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import mimetypes
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import soundfile as sf

from . import __version__
from .cli import Job, Session, _JobParser, build_parser, prepare_batch
from .config import user_config_path
from .effects import find_ffmpeg, process
from .engines import ENGINES

STATIC = Path(__file__).parent / "static"
MAX_BODY = 60 * 1024 * 1024
FORMATS = ("wav", "mp3", "ogg", "flac")


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.status = status
        self.extra = extra


@dataclass
class WebJob:
    id: str
    kind: str  # "speak", "preview" or "batch"
    label: str
    text: str
    status: str = "queued"  # queued, running, done, error
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    # Settings the job was made with, so the page can reuse them.
    settings: dict = field(default_factory=dict)
    run: Callable[[], dict] | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "text": self.text,
            "status": self.status,
            "log": self.log[-8:],
            "result": self.result,
            "error": self.error,
            "created": self.created,
            "settings": self.settings,
        }


class _JobLog(io.TextIOBase):
    """stdout replacement while a job runs: echo to the console and keep lines for the page."""

    def __init__(self, job: WebJob):
        self.job = job
        self.console = sys.__stdout__

    def write(self, s: str) -> int:
        if self.console:
            self.console.write(s)
        for line in s.splitlines():
            if line.strip():
                self.job.log.append(line.removeprefix("yapierre: ").strip())
        return len(s)

    def flush(self) -> None:
        if self.console:
            self.console.flush()


class App:
    def __init__(self, session: Session):
        self.session = session
        self.tmp = Path(tempfile.mkdtemp(prefix="yapierre-web-"))
        self.preview_dir = user_config_path().parent / "previews"
        self.files: dict[str, Path] = {}  # clip id -> file served at /api/audio/<id>
        self.raw: dict[str, tuple[Path, int]] = {}  # clip id -> audio before effects
        self.uploads: dict[str, Path] = {}
        self.jobs: dict[str, WebJob] = {}
        self.queue: queue.Queue[WebJob] = queue.Queue()
        self.current: WebJob | None = None
        self._device: str | None = None
        threading.Thread(target=self._worker, daemon=True).start()

    # --- jobs ----------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            self.current = job
            job.status = "running"
            try:
                with contextlib.redirect_stdout(_JobLog(job)):
                    job.result = job.run()
                job.status = "done"
            except SystemExit as e:
                job.error = e.code if isinstance(e.code, str) else "failed"
                job.status = "error"
            except Exception as e:  # noqa: BLE001 - report it on the page, keep serving
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"
            finally:
                job.run = None
                self.current = None

    def _submit(self, job: WebJob) -> WebJob:
        self.jobs[job.id] = job
        if job.run:
            self.queue.put(job)
        return job

    def _clip(self, path: Path, raw: Path | None, sample_rate: int, **meta) -> dict:
        clip_id = uuid.uuid4().hex[:12]
        self.files[clip_id] = path
        if raw:
            self.raw[clip_id] = (raw, sample_rate)
        return {
            "clip_id": clip_id,
            "url": f"/api/audio/{clip_id}",
            "filename": path.name,
            "path": str(path.resolve()),
            **meta,
        }

    def _run_job(self, job: Job, **meta) -> Callable[[], dict]:
        def run() -> dict:
            raw = self.tmp / f"raw_{uuid.uuid4().hex[:12]}.wav"
            result = self.session.run(job, raw_copy=raw)
            return self._clip(
                result.output,
                raw,
                result.sample_rate,
                duration=round(result.duration, 2),
                elapsed=round(result.elapsed, 1),
                effects=job.effects,
                loudness=job.loudness,
                **meta,
            )

        return run

    # --- request helpers -----------------------------------------------------------

    def _namespace(self, data: dict, *, preview: bool = False):
        """Turn the page's form fields into the same options the command line produces."""
        ns = build_parser(_JobParser).parse_args([])
        ns.text = [str(data.get("text") or "")]
        ns.lang = data.get("language") or None
        ns.engine = data.get("engine") or None
        ns.style = (data.get("style") or "").strip() or None
        ns.effect = [str(e) for e in data.get("effects") or []]
        ns.no_auto_effects = preview or not data.get("auto_effects", True)
        mode = data.get("mode", "design")
        if mode == "speaker":
            ns.speaker = data.get("speaker") or None
            if not ns.speaker:
                raise ApiError("Pick a preset speaker.")
        elif mode == "clone":
            upload = self.uploads.get(str(data.get("ref_upload") or ""))
            if not upload:
                raise ApiError("Upload or record a reference voice first.")
            ns.ref_audio = str(upload)
            ns.ref_text = (data.get("ref_text") or "").strip() or None
        else:
            ns.voice = (data.get("voice") or "").strip() or None
        seed = data.get("seed")
        ns.seed = int(seed) if seed not in (None, "") else None
        self._apply_loudness(ns, data)
        return ns

    @staticmethod
    def _apply_loudness(ns, data: dict) -> None:
        """Copy the page's normalize/loudness fields onto CLI-style options; missing means the config default."""
        if data.get("normalize") is False:
            ns.no_normalize = True
        elif data.get("loudness") not in (None, ""):
            try:
                ns.loudness = float(data["loudness"])
            except (TypeError, ValueError):
                raise ApiError("Loudness must be a number, e.g. -16.") from None

    def _target_loudness(self, data: dict) -> float | None:
        ns = argparse.Namespace(no_normalize=False, loudness=None)
        self._apply_loudness(ns, data)
        try:
            return self.session.target_loudness(ns)
        except SystemExit as e:
            raise ApiError(str(e.code).removeprefix("yapierre: ")) from None

    def _output_path(self, data: dict) -> Path:
        fmt = str(data.get("format") or self.session.config["output_format"]).lstrip(".").lower()
        if fmt not in FORMATS:
            raise ApiError(f"Unsupported format: {fmt}")
        name = Path(str(data.get("filename") or "")).name.strip()
        if not name:
            return self.session.default_output(fmt)
        if Path(name).suffix.lower().lstrip(".") not in FORMATS:
            name = f"{name}.{fmt}"
        return Path(self.session.config["output_dir"]) / name

    def _prepare(self, ns, label: str = "") -> Job:
        try:
            return self.session.prepare(ns, allow_stdin=False, label=label)
        except SystemExit as e:
            raise ApiError(str(e.code).removeprefix("yapierre: ")) from None

    # --- API -----------------------------------------------------------------------

    def info(self) -> dict:
        engine_name = self.session.config["engine"]
        engine = self.session.engine(engine_name)
        qwen = self.session.config.get("engines", {}).get(engine_name, {})
        return {
            "version": __version__,
            "engine": engine_name,
            "engines": [{"name": n, "description": d} for n, (_, d) in ENGINES.items()],
            "languages": engine.languages(),
            "speakers": engine.speakers(),
            "effects": [
                {"name": e.name, "description": e.description, "keywords": e.keywords}
                for e in self.session.effects.values()
            ],
            "formats": list(FORMATS),
            "defaults": {
                "language": self.session.config["language"],
                "format": self.session.config["output_format"],
                "voice": qwen.get("default_voice", ""),
                "auto_effects": self.session.config.get("auto_effects", True),
                "normalize": bool(self.session.config.get("normalize", True)),
                "loudness": float(self.session.config.get("loudness", -16.0)),
                "output_dir": str(Path(self.session.config["output_dir"]).resolve()),
            },
            "ffmpeg": bool(find_ffmpeg()),
        }

    def status(self) -> dict:
        if self._device is None:
            self._device = _describe_device(self.session.config.get("device", "auto"))
        loaded = []
        for engine in list(self.session._engines.values()):
            loaded += engine.loaded_models()
        current = self.current
        return {
            "device": self._device,
            "loaded_models": loaded,
            "busy": current is not None,
            "current": current.label if current else None,
            "queued": self.queue.qsize(),
        }

    def speak(self, data: dict) -> dict:
        ns = self._namespace(data)
        ns.output = str(self._output_path(data))
        job = self._prepare(ns)
        text = job.request.text
        web_job = WebJob(uuid.uuid4().hex[:12], "speak", _short(text), text, settings=data)
        web_job.run = self._run_job(job)
        return self._submit(web_job).to_json()

    def preview(self, data: dict) -> dict:
        """Short sample of a voice. Results are cached on disk, so repeat previews are instant."""
        data = {**data, "effects": [], "format": "wav"}
        if not (data.get("text") or "").strip():
            data["text"] = "Hello! This is what my voice sounds like."
        key_fields = {k: data.get(k) for k in ("mode", "speaker", "voice", "style", "language", "text", "engine")}
        key_fields["loudness"] = self._target_loudness(data)
        key = hashlib.sha1(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]
        path = self.preview_dir / f"{key}.wav"
        who = data.get("speaker") if data.get("mode") == "speaker" else (data.get("voice") or "default voice")
        label = f"Preview: {_short(who, 40)}"
        web_job = WebJob(uuid.uuid4().hex[:12], "preview", label, data["text"], settings=data)

        if path.exists():
            info = sf.info(str(path))
            web_job.status = "done"
            web_job.result = self._clip(path, path, info.samplerate, duration=round(info.duration, 2), cached=True)
            return self._submit(web_job).to_json()

        ns = self._namespace(data, preview=True)
        ns.seed = 0 if ns.seed is None else ns.seed  # same voice every time for the same description
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        ns.output = str(path)
        web_job.run = self._run_job(self._prepare(ns), cached=False)
        return self._submit(web_job).to_json()

    def batch(self, data: dict) -> dict:
        content = str(data.get("text") or "")
        defaults = build_parser(_JobParser).parse_args([])
        jobs, errors = prepare_batch(self.session, content, defaults)
        if errors:
            raise ApiError(
                f"{len(errors)} invalid line(s); nothing was started.",
                errors=[{"line": n, "message": m.removeprefix("yapierre: ")} for n, m in errors],
            )
        if not jobs:
            raise ApiError("No jobs: write one job per line.")
        created = []
        for job in jobs:
            text = job.request.text
            web_job = WebJob(uuid.uuid4().hex[:12], "batch", f"{job.label}: {_short(text, 40)}", text)
            web_job.run = self._run_job(job)
            created.append(self._submit(web_job).to_json())
        return {"jobs": created}

    def apply_effects(self, data: dict) -> dict:
        """Re-render an existing clip with different effects, without generating it again."""
        source = self.raw.get(str(data.get("clip_id") or ""))
        if not source:
            raise ApiError("That clip is no longer available; generate it again.", 404)
        if not find_ffmpeg():
            raise ApiError("ffmpeg is needed for effects but wasn't found on PATH.")
        names = [str(e) for e in data.get("effects") or []]
        unknown = [n for n in names if n not in self.session.effects]
        if unknown:
            raise ApiError(f"Unknown effect(s): {', '.join(unknown)}")
        raw, sample_rate = source
        loudness = self._target_loudness(data)
        out = self.tmp / f"fx_{uuid.uuid4().hex[:12]}.wav"
        started = time.time()
        try:
            process(
                raw, out, names, self.session.effects, sample_rate,
                loudness=loudness, true_peak=float(self.session.config.get("true_peak", -1.5)),
            )
        except SystemExit as e:
            raise ApiError(str(e.code)) from None
        info = sf.info(str(out))
        return self._clip(
            out, raw, sample_rate, duration=round(info.duration, 2), elapsed=round(time.time() - started, 2),
            effects=names, loudness=loudness,
        )

    def upload(self, body: bytes, filename: str) -> dict:
        """Store a reference recording for cloning, converted to mono WAV."""
        if not body:
            raise ApiError("Empty upload.")
        upload_id = uuid.uuid4().hex[:12]
        suffix = Path(filename).suffix.lower()[:8] or ".bin"
        src = self.tmp / f"upload_{upload_id}{suffix}"
        src.write_bytes(body)
        dst = self.tmp / f"ref_{upload_id}.wav"
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise ApiError(f"Couldn't read that audio file: {result.stderr.strip()[:300]}")
        else:
            try:
                data, sr = sf.read(str(src))
            except Exception:  # noqa: BLE001
                raise ApiError("Couldn't read that audio file (install ffmpeg for MP3/M4A/WebM).") from None
            sf.write(str(dst), data, sr)
        info = sf.info(str(dst))
        if info.duration < 1:
            raise ApiError("The recording is too short; use at least 3 seconds of speech.")
        self.uploads[upload_id] = dst
        clip_id = uuid.uuid4().hex[:12]
        self.files[clip_id] = dst
        return {
            "upload_id": upload_id,
            "duration": round(info.duration, 2),
            "url": f"/api/audio/{clip_id}",
            "path": str(dst),
        }


def _short(text: str, n: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _describe_device(setting: str) -> str:
    try:
        import torch
    except ImportError:
        return "unknown"
    if setting not in ("auto", "") and not setting.startswith("cuda"):
        return setting
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} ({props.total_memory / 1024**3:.0f} GB)"
    return "CPU (no CUDA GPU found)"


class Handler(BaseHTTPRequestHandler):
    app: App  # set by serve()
    server_version = f"yapierre/{__version__}"

    def log_message(self, format: str, *args: Any) -> None:
        pass  # keep the console for generation progress

    # --- responses -----------------------------------------------------------------

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, download: bool = False) -> None:
        if not path.is_file():
            self._json({"error": "File not found"}, 404)
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        status = 200
        # Range support lets the browser's audio player seek.
        if rng := self.headers.get("Range"):
            with contextlib.suppress(ValueError):
                first, last = rng.removeprefix("bytes=").split("-", 1)
                start = int(first) if first else max(0, size - int(last))
                end = min(int(last), size - 1) if first and last else size - 1
                status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError("Upload too large (60 MB max).", 413)
        return self.rfile.read(length) if length else b""

    def _handle(self, fn: Callable[[], Any]) -> None:
        try:
            self._json(fn())
        except ApiError as e:
            self._json({"error": str(e), **e.extra}, e.status)
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as e:  # noqa: BLE001 - return the error instead of dropping the connection
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    # --- routes --------------------------------------------------------------------

    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = url.path
        app = self.app
        if path in ("/", "/index.html"):
            self._file(STATIC / "index.html")
        elif path == "/api/info":
            self._handle(app.info)
        elif path == "/api/status":
            self._handle(app.status)
        elif path == "/api/jobs":
            self._handle(lambda: [j.to_json() for j in app.jobs.values()])
        elif path.startswith("/api/jobs/"):
            job = app.jobs.get(path.rsplit("/", 1)[-1])
            self._json(job.to_json()) if job else self._json({"error": "Unknown job"}, 404)
        elif path.startswith("/api/audio/"):
            file = app.files.get(path.rsplit("/", 1)[-1])
            download = "download" in parse_qs(url.query)
            self._file(file, download) if file else self._json({"error": "Unknown clip"}, 404)
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        app = self.app
        if path == "/api/upload":
            name = unquote(self.headers.get("X-Filename", "upload.bin"))
            self._handle(lambda: app.upload(self._body(), name))
            return
        routes = {
            "/api/speak": app.speak,
            "/api/preview": app.preview,
            "/api/batch": app.batch,
            "/api/effects": app.apply_effects,
        }
        if path not in routes:
            self._json({"error": "Not found"}, 404)
            return

        def call():
            try:
                data = json.loads(self._body() or b"{}")
            except json.JSONDecodeError:
                raise ApiError("Invalid JSON") from None
            return routes[path](data)

        self._handle(call)


def serve(session: Session, host: str, port: int, open_browser: bool = True) -> int:
    app = App(session)
    Handler.app = app
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        raise SystemExit(f"yapierre: can't listen on {host}:{port}: {e.strerror}. Try another --port.")
    server.daemon_threads = True
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown_host}:{port}/"
    print(f"yapierre: web interface running at {url}  (Ctrl+C to stop)")
    print(f"yapierre: clips are saved to {Path(session.config['output_dir']).resolve()}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("yapierre: warning: listening beyond this computer; anyone on your network can use it.")
    if not find_ffmpeg():
        print("yapierre: note: ffmpeg not found, so effects and MP3/OGG output won't work.")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nyapierre: stopping")
    finally:
        server.server_close()
        shutil.rmtree(app.tmp, ignore_errors=True)
    return 0
