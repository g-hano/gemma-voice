#!/usr/bin/env python
"""Train Gemma → Mimi speech head on Turkish speech–text pairs."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma_turkish.speech.config import load_config  # noqa: E402
from gemma_turkish.speech.model import GemmaSpeechModel  # noqa: E402
from gemma_turkish.speech.trainer import build_trainer  # noqa: E402
# Smaller backbone for local smoke without downloading E4B (~8B).

_SMOKE_DEFAULT_GEMMA = "google/gemma-2-2b-it"
# Ungated fallback when HF is not logged in (wiring / loss smoke only; not Gemma weights).
_DEV_PUBLIC_BACKBONE = "Qwen/Qwen2.5-0.5B-Instruct"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Turkish Gemma → Mimi speech-head training")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--demo", action="store_true", help="Use tiny synthetic Turkish set")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--gemma_model_id", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Demo data, 1 step; defaults to gemma-2-2b-it unless --gemma_model_id set",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Load YAML + print summary; no model weights downloaded",
    )
    p.add_argument(
        "--training-mode",
        type=str,
        choices=("teacher_forced", "generated_answer"),
        default=None,
        help="teacher_forced (repeat phrase) or generated_answer (Gemma generate → Mimi)",
    )
    p.add_argument(
        "--dev-backbone",
        action="store_true",
        help=f"Use public {_DEV_PUBLIC_BACKBONE} instead of gated Gemma (demo wiring only)",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    overrides: dict = {}
    if args.demo or args.smoke:
        overrides["use_demo_dataset"] = True
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.gemma_model_id:
        overrides["gemma_model_id"] = args.gemma_model_id
    if args.max_samples is not None:
        overrides["max_samples"] = args.max_samples
    if args.bf16 is not None:
        overrides["bf16"] = args.bf16
    if args.fp16 is not None:
        overrides["fp16"] = args.fp16
    if args.training_mode is not None:
        overrides["training_mode"] = args.training_mode
    if args.dev_backbone:
        overrides["gemma_model_id"] = _DEV_PUBLIC_BACKBONE
        overrides["head_hidden_dim"] = 896
        overrides["gradient_checkpointing"] = False

    cfg = load_config(args.config, overrides=overrides or None, project_root=ROOT)

    if args.validate_only:
        print("Config OK:")
        for k, v in sorted(cfg.to_dict().items()):
            print(f"  {k}: {v}")
        return

    if args.smoke:
        cfg.max_steps = 1
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 1
        cfg.eval_steps = 1
        cfg.save_steps = 0
        cfg.max_samples = min(cfg.max_samples or 4, 4)
        cfg.use_demo_dataset = True
        cfg.bf16 = False
        cfg.fp16 = False
        cfg.gemma_dtype = "float32"
        cfg.gradient_checkpointing = False

    if (
        (args.demo or args.smoke)
        and not args.dev_backbone
        and args.gemma_model_id is None
        and "gemma-4" in cfg.gemma_model_id.lower()
    ):
        print(
            f"Demo/smoke: using {_SMOKE_DEFAULT_GEMMA} "
            f"(pass --gemma_model_id google/gemma-4-E4B-it for full backbone)."
        )
        cfg.gemma_model_id = _SMOKE_DEFAULT_GEMMA

    if not torch.cuda.is_available():
        cfg.bf16 = False
        cfg.fp16 = False
        cfg.gemma_dtype = "float32"
        print("CUDA not available — using float32 on CPU (slow; prefer --demo --smoke).")

    print(
        f"Gemma: {cfg.gemma_model_id} | head: {cfg.head_type} | mode: {cfg.training_mode} | "
        f"Mimi: {cfg.mimi_model_id} | out: {cfg.output_dir}"
    )

    model = GemmaSpeechModel(cfg)
    trainer = build_trainer(model, cfg, smoke=args.smoke)
    trainer.train()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer_mix": model.layer_mix.state_dict(),
            "speech_head": model.speech_head.state_dict(),
            "config": model.config.to_dict(),
        },
        out / "speech_head.pt",
    )
    cfg.save_json(out / "speech_train_config.json")
    print(f"Checkpoint: {out / 'speech_head.pt'}")
    if cfg.training_mode == "generated_answer" and cfg.log_generated_outputs:
        print(f"Generated outputs: {cfg.generated_log_path}")
    print("Training finished.")

if __name__ == "__main__":
    main()