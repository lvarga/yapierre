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
- **Batch and interactive modes** that load the model once for many jobs
- **Web interface** with voice previews, live effect previews, microphone recording for cloning, and a batch editor
- **Loudness normalization** (EBU R128) so every clip plays at the same volume, -16 LUFS by default
- Output to WAV, MP3, OGG or FLAC; reads text from arguments, files or stdin; handles long texts

## Requirements

- Python 3.11 or 3.12
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH` (for effects, loudness normalization, MP3/OGG output and `--play`)
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

# Loudness: clips are normalized to -16 LUFS unless you say otherwise
yapierre "Breaking news!" --loudness -23        # broadcast level
yapierre "Breaking news!" --no-normalize        # keep the level the model produced

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

### Loudness

Every clip is normalized to the same perceived loudness after effects are applied, so a whisper,
a robot and a shouting voice all play at a similar volume. yapierre measures the clip (EBU R128),
applies the exact gain needed, and uses a peak limiter so nothing goes over -1.5 dBTP.

| Target | Typical use |
|---|---|
| `-14` LUFS | Streaming platforms, louder |
| `-16` LUFS | Online speech and podcasts (default) |
| `-23` LUFS | Broadcast (EBU R128) |

Set the target with `--loudness`, turn normalization off with `--no-normalize`, or change the defaults
with `normalize`, `loudness` and `true_peak` in the config. Normalization needs ffmpeg; without it,
clips keep their original level.

### Many jobs at once: batch and interactive mode

Loading a model takes 15-90 seconds, while generating a sentence takes a few. To make many clips,
run them in one session so the model loads only once.

**Batch mode** runs one job per line of a text file. Each line takes the same options as the command
line, without the `yapierre` part ([example](examples/jobs.txt)):

```text
# jobs.txt: lines starting with # are comments
"Welcome to the show!" --speaker Ryan -s excited -o intro.wav
"Bienvenue à l'émission !" -l fr --voice "a calm female narrator" -o intro_fr.mp3
"Resistance is futile." -s "sound like a robot" -o robot.wav
```

```bash
yapierre --batch jobs.txt
yapierre --batch jobs.txt -l de -e radio    # options here are defaults for every line
```

- Blank lines and lines starting with `#` are skipped.
- Lines without `-o` get an automatic name like `yapierre_<timestamp>.wav`.
- Relative paths (`-o`, `-f`, `--ref-audio`) are resolved from the folder you run yapierre in.
- All lines are checked before anything is generated, and jobs that use the same model run back to back.
- A failed job doesn't stop the batch; the summary at the end lists what failed.

**Interactive mode** keeps the model loaded and waits for you to type jobs:

```text
$ yapierre --interactive
yapierre> "Hello there" -s cheerful -o hi.wav
yapierre> Guten Morgen -l de --speaker Ryan
yapierre> quit
```

Quote text that contains `'` or `"`, or that starts with `-`. Options such as `--device` and `--config`
can only be given when starting yapierre. By default only one model is kept in memory, so switching
between `--voice`, `--speaker` and `--ref-audio` jobs reloads models. With enough VRAM, raise
`max_loaded_models` in the config (each model needs about 4.5 GB).

## Web interface

```bash
yapierre --serve
```

This starts a local server at <http://127.0.0.1:8765> and opens it in your browser. Models stay loaded
while it runs, and everything the command line can do is available on four tabs:

- **Speak**: text (typed or loaded from a .txt file), language, output format, loudness and voice, which can be
  described, a preset speaker, or cloned from an uploaded file or a microphone recording. Also style,
  effects, seed and file name. Once a clip is ready, click effect names under it to hear variations
  instantly without regenerating. **Copy as command** gives the equivalent `yapierre` command, and
  **Add to batch** turns the form into a batch line.
- **Voices**: hear every preset speaker and some voice-design ideas say a sample sentence, or describe
  your own voice. Previews are saved, so replaying one is instant.
- **Effects**: preview each effect, or a stack of them, on any clip.
- **Batch**: write or load a batch file, check it and run it.

Generated clips appear in the history panel with players, downloads and their file paths. They are
saved to `output_dir` like on the command line. The history lasts until the server stops.

Options: `--port 9000`, `--no-browser`, and `--host 0.0.0.0` to allow other devices on your network.
There is no login, so only do that on a network you trust.

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
loudness = -16.0               # LUFS; normalize = false keeps the generated level
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
