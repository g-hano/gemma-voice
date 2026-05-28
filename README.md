# Gemma Turkish — Gemma 4 speech head (Turkish)

Turkish-focused training stack for **[Gemma 4](https://huggingface.co/google/gemma-4-E2B-it)** with a **Frisson-style** frozen-backbone → **Mimi** speech-token head ([architecture blog](https://www.frisson-labs.com/gemma4-e4b-architecture)).

## Quick start

```bash
git clone https://github.com/g-hano/gemma.git
cd gemma
pip install -e .
python scripts/train_speech.py --config configs/speech_default.yaml
```

Requires **Python 3.11+** and a CUDA-capable GPU for training. PyTorch is installed via `pyproject.toml` dependencies.

Optional checks:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
huggingface-cli login   # if using a gated Gemma checkpoint
```

Validate config and data loading before a full run:

```bash
python scripts/train_speech.py --config configs/speech_default.yaml --validate-only
```

Smoke test (no backbone download):

```bash
python scripts/train_speech.py --demo --smoke
```

See **[docs/training.md](docs/training.md)** for dataset columns, VRAM notes, and commands.

## Defaults

| Item | Value |
|------|--------|
| Backbone | `google/gemma-4-E2B-it` |
| Codec | `kyutai/mimi` (8 codebooks, 24 kHz) |
| Dataset | `Anilosan15/Synthetic_Turkish_TTS_Data` (CC BY 4.0) |
| Tap | Last **6** decoder layers |
| Hidden | **2560** |

Override the backbone in config or on the CLI, e.g. `google/gemma-4-E4B-it` for the full E4B model.

## RTX 3080 (12 GB)

- Use **bf16**, **batch size 1**, **gradient_checkpointing: true** (defaults in `configs/speech_default.yaml`)
- E2B + Mimi fits more comfortably than E4B; reduce `max_audio_seconds` or use a larger GPU if OOM
- For CUDA 12.4 wheels on Windows, you may reinstall PyTorch explicitly:
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

## Documentation

- **[docs/setup.md](docs/setup.md)** — Research notes, Turkish text datasets, environment
- **[docs/training.md](docs/training.md)** — Speech-head training

## References

- [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- [Frisson Labs — Grafting a Speech Head onto Gemma 4 E4B](https://www.frisson-labs.com/gemma4-e4b-architecture)
- [gemma4-audio](https://github.com/frisson-labs/gemma4-audio)
