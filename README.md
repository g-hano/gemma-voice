# Gemma Turkish — Gemma 4 E4B speech head (Turkish)

Turkish-focused training stack for **[Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it)** with a **Frisson-style** frozen-backbone → **Mimi** speech-token head ([architecture blog](https://www.frisson-labs.com/gemma4-e4b-architecture)).

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\Cihan\Desktop\gemma

uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1

uv pip install -e ".[logging]"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU (RTX 3080):

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Hugging Face (gated Gemma, if used):

```powershell
huggingface-cli login
```

## Speech-head training

```powershell
python scripts/train_speech.py --config configs/speech_default.yaml --validate-only
python scripts/train_speech.py --config configs/speech_default.yaml
```

Smoke (no E4B download):

```powershell
python scripts/train_speech.py --demo --smoke
```

See **[docs/training.md](docs/training.md)** for dataset columns, VRAM notes, and commands.

## Defaults

| Item | Value |
|------|--------|
| Backbone | `google/gemma-4-E4B-it` |
| Codec | `kyutai/mimi` (8 codebooks, 24 kHz) |
| Dataset | `Anilosan15/Synthetic_Turkish_TTS_Data` (CC BY 4.0) |
| Tap | Last **6** of **42** decoder layers |
| Hidden | **2560** (E4B text decoder) |

## RTX 3080 (12 GB)

- Use **bf16**, **batch size 1**, **gradient_checkpointing: true**
- E4B + Mimi is **tight** on 12 GB; reduce `max_audio_seconds` or use a larger GPU if OOM
- Install **CUDA 12.4** PyTorch wheels (`cu124`) as above

## Documentation

- **[docs/setup.md](docs/setup.md)** — Research notes, Turkish text datasets, environment
- **[docs/training.md](docs/training.md)** — Speech-head training

## References

- [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- [Frisson Labs — Grafting a Speech Head onto Gemma 4 E4B](https://www.frisson-labs.com/gemma4-e4b-architecture)
- [gemma4-audio](https://github.com/frisson-labs/gemma4-audio)
