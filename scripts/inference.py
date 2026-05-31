#!/usr/bin/env python
"""Turkish TTS inference with a trained Gemma → Mimi speech-head checkpoint."""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "outputs" / "speech_head_v2" / "checkpoint-12000"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma_turkish.speech.hub import GemmaTurkishTTS  # noqa: E402
from gemma_turkish.speech.model import estimate_speech_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesize Turkish speech from text using a speech-head checkpoint",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"HF repo id, checkpoint folder, or speech_head.pt (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument("--text", "-t", type=str, default=None, help="Text to speak")
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output WAV path (default: outputs/inference/<timestamp>.wav)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Max Mimi frames (default: estimate from text length)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0 = argmax; default: from checkpoint config)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top-p (default: from checkpoint config)",
    )
    p.add_argument("--argmax", action="store_true", help="Greedy argmax decoding (temperature=0)")
    p.add_argument("--no-trim", action="store_true", help="Keep trailing silence in output")
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: enter text lines until empty input",
    )
    return p.parse_args()


def load_model(checkpoint: str | Path) -> tuple[GemmaTurkishTTS, int]:
    ckpt = str(checkpoint)
    print(f"Loading: {ckpt}")
    model = GemmaTurkishTTS.from_pretrained(ckpt)
    step = getattr(model, "_hub_step", 0)
    device = next(model.parameters()).device
    print(f"Ready (step {step}) on {device} — Gemma + Mimi + speech head merged.")
    return model, step


def synthesize_one(
    model: GemmaTurkishTTS,
    text: str,
    *,
    output: Path | None,
    num_frames: int | None,
    temperature: float | None,
    top_p: float | None,
    argmax: bool,
    trim: bool,
) -> Path:
    text = text.strip()
    if not text:
        raise ValueError("Empty text.")

    if argmax:
        model.config.synth_temperature = 0.0
    elif temperature is not None:
        model.config.synth_temperature = temperature
    if top_p is not None:
        model.config.synth_top_p = top_p

    n = num_frames or estimate_speech_frames(text, model.config)
    print(f"Synthesizing ({n} frames max): {text[:80]}{'...' if len(text) > 80 else ''}")

    wave = model.synthesize(text, num_frames=n, trim_silence=trim)
    audio = wave.squeeze().numpy()
    sr = model.config.mimi_sample_rate
    dur = len(audio) / sr

    if output is None:
        out_dir = ROOT / "outputs" / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = out_dir / f"{stamp}.wav"
    else:
        output = Path(output)
        if not output.is_absolute():
            output = (ROOT / output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(output), audio, sr)
    output.with_suffix(".txt").write_text(
        f"{text}\n\nframes={n} duration={dur:.2f}s "
        f"temp={model.config.synth_temperature} top_p={model.config.synth_top_p}",
        encoding="utf-8",
    )
    print(f"Saved {dur:.2f}s -> {output}")
    return output


def main() -> None:
    args = parse_args()
    model, _step = load_model(args.checkpoint)
    trim = not args.no_trim

    if args.interactive or not args.text:
        print("Interactive mode — empty line to quit.")
        while True:
            try:
                line = input("Metin> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                break
            synthesize_one(
                model,
                line,
                output=None,
                num_frames=args.frames,
                temperature=args.temperature,
                top_p=args.top_p,
                argmax=args.argmax,
                trim=trim,
            )
        return

    synthesize_one(
        model,
        args.text,
        output=Path(args.output) if args.output else None,
        num_frames=args.frames,
        temperature=args.temperature,
        top_p=args.top_p,
        argmax=args.argmax,
        trim=trim,
    )


if __name__ == "__main__":
    main()
