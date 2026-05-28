#!/usr/bin/env python
"""Load a speech-head checkpoint and predict Mimi tokens for one Turkish prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma_turkish.speech.config import SpeechTrainConfig, load_config  # noqa: E402
from gemma_turkish.speech.model import GemmaSpeechModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Trainer output dir")
    p.add_argument("--text", type=str, default="Merhaba, bu bir deneme cümlesidir.")
    p.add_argument("--num_frames", type=int, default=25, help="Mimi frames to predict (~2s @ 12.5 Hz)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--decode_wav", type=str, default=None, help="Optional output WAV path")
    return p.parse_args()


def load_head_weights(model: GemmaSpeechModel, checkpoint_dir: Path) -> None:
    head_path = checkpoint_dir / "speech_head.pt"
    if not head_path.is_file():
        raise FileNotFoundError(f"Missing {head_path}; train with scripts/train_speech.py first.")
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    model.layer_mix.load_state_dict(state["layer_mix"])
    model.speech_head.load_state_dict(state["speech_head"])
    if "config" in state:
        for key, value in state["config"].items():
            if hasattr(model.config, key):
                setattr(model.config, key, value)


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    cfg_path = args.config or ckpt / "speech_train_config.json"
    if Path(cfg_path).suffix == ".json":
        import json

        with Path(cfg_path).open(encoding="utf-8") as f:
            cfg = SpeechTrainConfig.from_dict(json.load(f))
    else:
        cfg = load_config(cfg_path)
    cfg.resolve_paths(ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GemmaSpeechModel(cfg).to(device)
    load_head_weights(model, ckpt)
    model.eval()

    encoded = model.encode_text_prompt(args.text)
    batch = model.tokenizer.pad(
        {"input_ids": [encoded["input_ids"]], "attention_mask": [encoded["attention_mask"]]},
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    codes = model.predict_speech_tokens(
        batch["input_ids"],
        batch["attention_mask"],
        num_frames=args.num_frames,
    )
    print(f"Predicted speech codes shape: {tuple(codes.shape)} (B, codebooks, frames)")

    if args.decode_wav:
        wav = model.codec.decode_codes(codes.to(device))
        import soundfile as sf

        out_path = Path(args.decode_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio = wav.squeeze().float().cpu().numpy()
        sf.write(out_path, audio, model.config.mimi_sample_rate)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
