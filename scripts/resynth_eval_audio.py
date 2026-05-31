#!/usr/bin/env python
"""Re-synthesize eval WAVs with correct duration (reference frames + trim, not fixed 12 s)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma_turkish.speech.config import load_config  # noqa: E402
from gemma_turkish.speech.data import (  # noqa: E402
    TurkishSpeechDataset,
    load_turkish_speech_dataset,
    train_val_split,
)
from gemma_turkish.speech.model import GemmaSpeechModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-synthesize eval audio from a speech-head checkpoint")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint folder or speech_head.pt (e.g. outputs/speech_head/checkpoint-4500)",
    )
    p.add_argument("--config", type=str, default="configs/speech_default.yaml")
    p.add_argument("--output-dir", type=str, default=None, help="WAV output dir")
    p.add_argument("--samples", type=int, default=2, help="How many eval rows to synthesize")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, project_root=ROOT)
    out_dir = Path(args.output_dir or (Path(cfg.output_dir) / "eval_audio_resynth"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model + checkpoint: {args.checkpoint}")
    model = GemmaSpeechModel(cfg)
    step = GemmaSpeechModel.load_trainable_checkpoint(model, Path(args.checkpoint))
    if torch.cuda.is_available():
        model = model.cuda()

    full = load_turkish_speech_dataset(cfg)
    _, eval_ds = train_val_split(full, cfg.val_fraction, cfg.seed)
    eval_set = TurkishSpeechDataset(
        eval_ds, model.encode_text_prompt, model.codec, cfg, split_label="eval"
    )

    sr = cfg.mimi_sample_rate
    tag = f"step{step:06d}_resynth"
    for i in range(min(args.samples, len(eval_set))):
        item = eval_set[i]
        text = item["text"]
        n = min(
            int(item["num_frames"]) + cfg.eval_audio_frame_margin,
            cfg.eval_audio_max_frames,
        )
        wave = model.synthesize(text, num_frames=n)
        audio = wave.squeeze().numpy()
        dur = len(audio) / sr
        wav_path = out_dir / f"{tag}_sample{i}.wav"
        sf.write(str(wav_path), audio, sr)
        (out_dir / f"{tag}_sample{i}.txt").write_text(
            f"{text}\n\nref_frames={item['num_frames']} gen_frames={n} duration={dur:.2f}s",
            encoding="utf-8",
        )
        print(f"  sample {i}: ref={item['num_frames']} gen={n} -> {dur:.2f}s -> {wav_path}")


if __name__ == "__main__":
    main()
