# Speech-head training (Turkish)

Train a **Frisson-style** adapter: frozen **Gemma 4 E4B-it** decoder → learned mix of the **last 6 layers** → parallel **Mimi** codec-token head. Default mode **`generated_answer`**: Gemma **autoregressively generates** a Turkish answer, then the speech head reads **mean-pooled hidden states over the generated answer tokens** and predicts **8 Mimi codebooks** at ~12.5 Hz (teacher WAV resampled to **24 kHz**). Legacy **`teacher_forced`** repeats a provided phrase via the last prompt token only.

Architecture: [Frisson blog](https://www.frisson-labs.com/gemma4-e4b-architecture) / [gemma4-audio](https://github.com/frisson-labs/gemma4-audio). Gemma and Mimi stay frozen; only the speech head trains.

## Prerequisites

1. **uv env** (see [setup.md](setup.md)):
   ```powershell
   cd C:\Users\Cihan\Desktop\gemma
   .\.venv\Scripts\Activate.ps1
   uv pip install -e ".[logging]"
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
2. **CUDA** (RTX 3080): verify with:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
3. **Hugging Face login** (Gemma 4 + Common Voice):
   ```powershell
   huggingface-cli login
   ```
   Accept the [Gemma license](https://huggingface.co/google/gemma-4-E4B-it) and [Common Voice terms](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) in the browser.
4. **Disk**: E4B (~8B bf16) + Mimi ≈ 16 GB weights; token cache under `data/speech_token_cache/`.

## Training modes

| Mode | Config | Behavior |
|------|--------|----------|
| **`generated_answer`** (default) | `training_mode: generated_answer` | User question → `model.generate()` → forward on full sequence → **mean-pool answer-span hiddens** → Mimi CE vs teacher WAV |
| **`teacher_forced`** (legacy) | `training_mode: teacher_forced` | Repeat prompt `Bunu doğal bir konuşma gibi söyle:\n{text}` → **last token** hidden → Mimi CE |

Generation prompt (Turkish):

```text
Şunu kısa ve Türkçe yanıtla:
{question}
```

Teacher-forced template (legacy):

```text
Bunu doğal bir konuşma gibi söyle:
{text}
```

Gemma 4 uses `AutoProcessor.apply_chat_template(..., enable_thinking=False)` for tokenization.

## Datasets & column mapping

| Role | HF path | Config keys | Columns used | Loader filters |
|------|---------|-------------|--------------|----------------|
| **Speech head (default)** | `mozilla-foundation/common_voice_17_0` | `dataset_config: tr`, `dataset_split: train` | `sentence` → text, `audio` → waveform | `min_text_chars` 4, `max_text_chars` 500, audio 0.5–30 s, resample 24 kHz |
| **Smoke / offline** | `--demo` | synthetic | `question`, `sentence`, `audio` | 4 Turkish QA pairs + noise WAV |
| **Text SFT (phase A)** | `tascib/turkish-instruction`, etc. | — | — | See [setup.md](setup.md); not used by speech script |

Common Voice row shape (after `cast_column` to `Audio(24000)`):

| Column | Maps to |
|--------|---------|
| `sentence` | Turkish transcript → `text_prompt_template` |
| `audio` | `{"array", "sampling_rate"}` → Mimi teacher encode |
| `path` | Ignored (decoded via `audio`) |

## Run training

```powershell
cd C:\Users\Cihan\Desktop\gemma
.\.venv\Scripts\Activate.ps1

# Config check (no weight download)
python scripts/train_speech.py --config configs/speech_default.yaml --validate-only

# Full run (E4B + Common Voice TR)
python scripts/train_speech.py --config configs/speech_default.yaml

# Small generated-answer run (demo QA, 20 steps; auto-fallback to gemma-2-2b-it on --demo)
python scripts/train_speech.py --config configs/speech_default.yaml --max_steps 20 --demo

# Smoke: 1 step, float32, gemma-2-2b-it
python scripts/train_speech.py --demo --smoke

# Legacy repeat-phrase mode
python scripts/train_speech.py --demo --smoke --training-mode teacher_forced

# Full E4B (large download; tight on 12 GB VRAM)
python scripts/train_speech.py --config configs/speech_default.yaml --demo --gemma_model_id google/gemma-4-E4B-it
```

Checkpoints: `outputs/speech_head/` plus `speech_head.pt` (adapter weights only).

## Inference

```powershell
python scripts/generate_speech_tokens.py `
  --checkpoint outputs/speech_head `
  --text "Küçük adaptör donmuş katmanlar üzerinden net konuşmayı koşullar." `
  --num_frames 25 `
  --decode_wav outputs/sample.wav
```

## GPU memory — RTX 3080 (12 GB)

| Setting | Recommendation |
|---------|----------------|
| `gemma_dtype` | `bfloat16` |
| `gradient_checkpointing` | `true` (default) |
| `per_device_train_batch_size` | `1` |
| `gradient_accumulation_steps` | `8` (effective batch 8) |

| Component | VRAM (bf16, bs=1, checkpointing) |
|-----------|----------------------------------|
| Gemma 4 E4B frozen | ~10–11 GB |
| Mimi teacher encode | ~0.5 GB |
| Speech head + activations | ~0.5–1 GB |
| **Total** | **Often tight on 12 GB** — OOM possible |

Mitigations: shorter `max_audio_seconds`, smaller `max_samples` while debugging, 4-bit Gemma (not in this repo), or a 24 GB GPU.

## Import check

```powershell
python -c "from gemma_turkish.speech.config import SpeechTrainConfig; from gemma_turkish.speech import model, codec, data, collator, trainer; print('ok')"
```

## Layout

```
src/gemma_turkish/speech/
  config.py      # SpeechTrainConfig + YAML load
  codec.py       # Mimi encode/decode
  model.py       # GemmaSpeechModel (Gemma4ForConditionalGeneration)
  data.py        # TurkishSpeechDataset
  collator.py
  trainer.py
scripts/train_speech.py
scripts/generate_speech_tokens.py
configs/speech_default.yaml
```

## Gaps vs Frisson blog

| Item | Status |
|------|--------|
| E4B backbone + last-6 tap | Implemented |
| 8 codebooks, parallel CE | Implemented |
| ~152M head at hidden 2560 | Scaled via `head_hidden_dim: 2560` + efficient frame biases |
| Audio-in path (WAV-only prompt) | Not implemented |
| `generated_answer` training | Implemented (`forward_generated_answer`) |
| TTS teacher on generated text | v1 uses **reference sentence WAV** as Mimi target (proxy) |
| Temporal / streaming audio LM | Not implemented (optional follow-up flag) |
