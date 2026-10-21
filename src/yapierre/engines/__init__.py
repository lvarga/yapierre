"""Engine registry. Engines are imported lazily so heavy dependencies load only when used.

To add an engine: implement `yapierre.engines.base.Engine` and register it here.
"""

from __future__ import annotations

import importlib

from .base import Engine, SynthesisRequest

ENGINES: dict[str, tuple[str, str]] = {
    # name: (module:class, short description)
    "qwen": ("yapierre.engines.qwen:QwenEngine", "Qwen3-TTS: 10 languages, voice design, style instructions, cloning"),
}


def load_engine(name: str) -> type[Engine]:
    if name not in ENGINES:
        raise SystemExit(f"yapierre: unknown engine '{name}'. Available: {', '.join(ENGINES)}")
    target = ENGINES[name][0]
    module_name, class_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(
            f"yapierre: engine '{name}' is missing dependencies ({e}).\n"
            f"Install them with: pip install \"yapierre[{name}]\""
        )
    return getattr(module, class_name)


__all__ = ["ENGINES", "Engine", "SynthesisRequest", "load_engine"]
