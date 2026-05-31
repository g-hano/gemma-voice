#!/usr/bin/env python
"""Export merged speech-head checkpoint for Hugging Face Hub upload."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "outputs" / "speech_head_v2" / "checkpoint-12000"
DEFAULT_OUT = ROOT / "hf_export" / "gemma-turkish-speech-e2b-mimi"
SRC_PKG = ROOT / "src" / "gemma_turkish"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export speech-head for Hugging Face Hub")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--repo-id", type=str, default="Chan-Y/gemma-turkish-speech-e2b-mimi")
    return p.parse_args()


README_TEMPLATE = """---
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
- **Training steps:** {steps}

## Quick start

```bash
pip install torch transformers accelerate soundfile huggingface_hub
huggingface-cli login   # Gemma 4 is gated — accept license on HF first
```

**One line loads Gemma + Mimi + trained speech head** (backbone/codec pulled from official repos; only ~520 MB adapter weights in this repo):

```python
from gemma_turkish.speech.hub import GemmaTurkishTTS
import soundfile as sf

model = GemmaTurkishTTS.from_pretrained("{repo_id}")
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
"""


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (ROOT / ckpt).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (ROOT / out).resolve()

    pt_src = ckpt / "speech_head.pt" if ckpt.is_dir() else ckpt
    if not pt_src.is_file():
        sys.exit(f"Missing {pt_src}")

    state = __import__("torch").load(pt_src, map_location="cpu", weights_only=True)
    cfg_dict = state.get("config")
    if cfg_dict is None:
        fallback = ckpt.parent / "speech_train_config.json" if ckpt.is_dir() else ckpt.parent.parent / "speech_train_config.json"
        cfg_dict = json.loads(fallback.read_text(encoding="utf-8"))

    step = 0
    ts = ckpt / "trainer_state.json" if ckpt.is_dir() else None
    if ts and ts.is_file():
        step = int(json.loads(ts.read_text(encoding="utf-8")).get("global_step", 0))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "src").mkdir()

    shutil.copy2(pt_src, out / "speech_head.pt")
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")
    shutil.copytree(SRC_PKG, out / "src" / "gemma_turkish")

    (out / "requirements.txt").write_text(
        "torch>=2.1\n"
        "transformers>=4.45\n"
        "accelerate>=0.33\n"
        "soundfile>=0.12\n"
        "huggingface_hub>=0.23\n"
        "pyyaml>=6.0\n"
        "datasets>=2.19\n"
        "librosa>=0.10\n",
        encoding="utf-8",
    )

    (out / "README.md").write_text(
        README_TEMPLATE.format(steps=step, repo_id=args.repo_id),
        encoding="utf-8",
    )

    # HF-friendly inference entrypoint (loads weights from repo root)
    (out / "inference.py").write_text(
        '''#!/usr/bin/env python
"""Inference for HF repo — run from the downloaded model folder."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from gemma_turkish.speech.hub import GemmaTurkishTTS
from gemma_turkish.speech.model import estimate_speech_frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--text", required=True)
    p.add_argument("-o", "--output", default="output.wav")
    p.add_argument("--argmax", action="store_true")
    args = p.parse_args()

    model = GemmaTurkishTTS.from_pretrained(REPO)
    if args.argmax:
        model.config.synth_temperature = 0.0

    text = args.text.strip()
    n = estimate_speech_frames(text, model.config)
    wave = model.synthesize(text, num_frames=n)
    sf.write(args.output, wave.squeeze().numpy(), model.config.mimi_sample_rate)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )

    meta = {
        "repo_id": args.repo_id,
        "global_step": step,
        "base_model": cfg_dict.get("gemma_model_id"),
        "codec": cfg_dict.get("mimi_model_id"),
        "head_type": cfg_dict.get("head_type"),
    }
    (out / "model_index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported step {step} -> {out}")
    print(f"Upload: huggingface-cli upload {args.repo_id} {out} . --repo-type model")


if __name__ == "__main__":
    main()
