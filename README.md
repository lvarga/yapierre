# yapierre

**A command-line text-to-speech tool that yaps in many languages.**

yapierre turns text into speech with open AI voice models, running locally on your own machine.
You describe *how* it should sound in plain words, and it does the rest:

```bash
yapierre "I am your new overlord. Resistance is futile." --style "make it sound like a robot"
```

- **10 languages** out of the box: English, German, French, Spanish, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- **Style instructions** in plain language: `--style "whispering, scared"`, `--style "excited sports commentator"`
- **Voice design**: describe the speaker, e.g. `--voice "an old pirate with a gravelly voice"`
- **Preset speakers** and **voice cloning** from a short recording
- **Audio effects** like robot, radio, telephone, cave, chipmunk, giant, alien and underwater, plus your own ffmpeg filters
- **Pluggable engines**: ships with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS); more engines can be added
- Output to WAV, MP3, OGG or FLAC; reads text from arguments, files or stdin; handles long texts

## Requirements

- Python 3.11 or 3.12
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH` (for effects, MP3/OGG output and `--play`)
- An NVIDIA GPU with **6 GB+ VRAM** is strongly recommended (the default model uses about 4.5 GB).
  CPU and Apple Silicon (`mps`) work too, just much slower.
- About 4 GB of disk space per model; models download automatically on first use

## Installation

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) picks up the CUDA build of PyTorch automatically:

```bash
git clone https://github.com/lvarga/yapierre.git
cd yapierre
uv sync --extra qwen
uv run yapierre "Hello world"
```

To install it as a global `yapierre` command:

```bash
uv tool install ".[qwen]"
```

### With pip

Install PyTorch for your hardware first (see [pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
pip install "yapierre[qwen] @ git+https://github.com/lvarga/yapierre.git"
```

## Usage

```bash
# Basic
yapierre "Hello world"                           # saves yapierre_<timestamp>.wav
yapierre "Hello world" -o hello.mp3 --play       # choose the file, then play it

# Languages (auto-detected by default)
yapierre "Guten Tag! Wie geht's?" -l de
yapierre "Bonjour tout le monde" -l french

# Describe the voice and the delivery
yapierre "Once upon a time..." \
  --voice "an old man with a warm, raspy voice" \
  --style "slow and mysterious"

# Preset speakers (see: yapierre --list speakers)
yapierre "Breaking news!" --speaker Ryan --style "very excited"

# Clone a voice from a 3-10 second recording; the transcript improves quality
yapierre "Anything you want" --ref-audio me.wav --ref-text "What I said in me.wav"

# Effects (see: yapierre --list effects); stack them by repeating -e
yapierre "Houston, we have a problem." -e radio
yapierre "Fee-fi-fo-fum" -e giant -e cave

# Long text from a file or a pipe
yapierre -f chapter1.txt -o chapter1.mp3
cat notes.txt | yapierre - -o notes.wav
```

### How the voice is chosen

| You pass | Model used | What it does |
|---|---|---|
| nothing, or `--voice` | VoiceDesign | Builds a voice from your description (or the configured default) |
| `--speaker NAME` | CustomVoice | Uses a preset speaker; `--style` still applies |
| `--ref-audio FILE` | Base | Clones the voice in the recording; `--style` is ignored |

`--style` is always passed to the model as an instruction. If it also mentions an effect keyword
(e.g. "robot", "radio", "underwater", "cave"), the matching effect is applied too, since the model alone
produces human voices. Turn that off with `--no-auto-effects` or `auto_effects = false`.

### Effects

| Effect | Sound |
|---|---|
| `robot` | Flat, metallic robot voice |
| `whisper` | Breathy, unvoiced whisper |
| `radio` | Old AM radio |
| `telephone` | Narrow-band phone call |
| `echo` | Short echo |
| `cave` | Big reverberant space |
| `chipmunk` | High-pitched cartoon voice |
| `giant` | Very deep, monstrous voice |
| `alien` | Warbling sci-fi alien |
| `underwater` | Muffled and wobbly |

## Configuration

Create a starter config with:

```bash
yapierre --init-config
```

This writes `%APPDATA%\yapierre\config.toml` on Windows or `~/.config/yapierre/config.toml` elsewhere.
A `yapierre.toml` in the current folder, or a file passed with `--config`, overrides it, and command-line
flags override everything. See [yapierre.example.toml](yapierre.example.toml) for all options. The main ones:

```toml
language = "auto"
output_format = "mp3"
cache_dir = "D:/models"        # where models are downloaded (default: ~/.cache/huggingface)
device = "auto"                # auto, cuda, cpu, mps

[engines.qwen]
default_voice = "A calm British narrator with a deep voice."

[engines.qwen.models]          # swap in the smaller 0.6B models for speed or low VRAM
custom = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

[effects.stadium]              # add your own ffmpeg audio filter chains
filter = "aecho=0.8:0.85:250|500:0.4|0.3,volume=1.2"
keywords = ["stadium", "arena"]
```

## Adding an engine

Engines live in [`src/yapierre/engines/`](src/yapierre/engines/). To add one:

1. Subclass `Engine` from [`base.py`](src/yapierre/engines/base.py) and implement `synthesize()`,
   which takes a `SynthesisRequest` and returns `(waveform, sample_rate)`.
2. Register it in `ENGINES` in [`engines/__init__.py`](src/yapierre/engines/__init__.py).
3. Add its dependencies as an optional extra in `pyproject.toml`.

Its settings come from the `[engines.<name>]` table in the config.

## Licenses

yapierre itself is released under the [Apache License 2.0](LICENSE): you can use, modify and
distribute it, personally or commercially.

**The AI models have their own licenses.** The default Qwen3-TTS models are also Apache 2.0, but
check the license of any other model you configure before using it commercially.

Voice cloning is powerful. Only clone voices you have permission to use, and don't use generated
speech to deceive people.
