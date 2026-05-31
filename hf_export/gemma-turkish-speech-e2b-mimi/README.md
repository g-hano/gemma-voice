---
license: gemma
base_model: google/gemma-4-E2B-it
tags:
  - text-to-speech
  - turkish
  - gemma
  - mimi
  - speech-synthesis
language:
  - tr
library_name: pytorch
pipeline_tag: text-to-speech
---

# Gemma Turkish Speech Head (E2B + Mimi)

Turkish TTS speech adapter for [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) with [kyutai/mimi](https://huggingface.co/kyutai/mimi) neural codec.

Trained on [Synthetic_Turkish_TTS_Data](https://huggingface.co/datasets/Anilosan15/Synthetic_Turkish_TTS_Data) (CC BY 4.0).

## Architecture

- **Frozen backbone:** Gemma 4 E2B-it (text conditioning)
- **Frozen codec:** Kyutai Mimi (8 codebooks @ 12.5 Hz, 24 kHz)
- **Trainable:** learned layer-mix (last 6 Gemma layers) + autoregressive cross-attention speech decoder
- **Training steps:** 12000

## Quick start

```bash
pip install torch transformers accelerate soundfile huggingface_hub
huggingface-cli login   # Gemma 4 is gated — accept license on HF first
```

**One line loads Gemma + Mimi + trained speech head** (backbone/codec pulled from official repos; only ~520 MB adapter weights in this repo):

```python
from gemma_turkish.speech.hub import GemmaTurkishTTS
import soundfile as sf

model = GemmaTurkishTTS.from_pretrained("Chan-Y/gemma-turkish-speech-e2b-mimi")
model = model.to("cuda")

text = "Merhaba, bu bir Türkçe ses sentezi denemesidir."
wave = model.synthesize(text)
sf.write("out.wav", wave.squeeze().numpy(), model.config.mimi_sample_rate)
```

Or use the bundled CLI after downloading this repo:

```bash
python inference.py -t "Merhaba dünya."
```

### Why not one giant checkpoint?

Gemma 4 E2B (~16 GB) and Mimi are **gated / upstream** models. Duplicating them in this repo would be slow to upload, hard to license, and stale when Google/Kyutai update weights. `GemmaTurkishTTS.from_pretrained` downloads **this repo's adapters** plus **official** `google/gemma-4-E2B-it` and `kyutai/mimi` at runtime — same UX as a merged model, without redundant storage.

## Files

| File | Description |
|------|-------------|
| `speech_head.pt` | Merged trainable weights (`layer_mix` + `speech_head`) + embedded config |
| `config.json` | Full training/inference hyperparameters |
| `src/gemma_turkish/` | Model loading & synthesis code |

## License

- Speech-head weights: same terms as base Gemma model (see Google Gemma license).
- Training data: CC BY 4.0 (Synthetic Turkish TTS dataset).
- Mimi codec: Kyutai license.
