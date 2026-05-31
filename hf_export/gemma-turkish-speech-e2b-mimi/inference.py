#!/usr/bin/env python
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
