"""Plot training metrics from HuggingFace trainer_state.json log_history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_log_history(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("log_history", [])
    train = [entry for entry in history if "loss" in entry]
    evals = [entry for entry in history if "eval_runtime" in entry]
    return train, evals


def plot_metric(
    train: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    color: str,
) -> None:
    steps = [entry["step"] for entry in train]
    values = [entry[metric] for entry in train]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, color=color, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined(train: list[dict], out_path: Path) -> None:
    steps = [entry["step"] for entry in train]
    loss = [entry["loss"] for entry in train]
    lr = [entry["learning_rate"] for entry in train]
    grad_norm = [entry["grad_norm"] for entry in train]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, loss, color="#2563eb", linewidth=1.2)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, lr, color="#16a34a", linewidth=1.2)
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, grad_norm, color="#dc2626", linewidth=1.2)
    axes[2].set_ylabel("Grad Norm")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Gradient Norm")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Speech Head Training Metrics (step 9000)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trainer_state",
        type=Path,
        help="Path to trainer_state.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plot PNG files (default: <checkpoint>/training_plots)",
    )
    args = parser.parse_args()

    train, _ = load_log_history(args.trainer_state)
    if not train:
        raise SystemExit("No training entries found in log_history.")

    out_dir = args.output_dir or (args.trainer_state.parent / "training_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        train,
        metric="loss",
        ylabel="Loss",
        title="Training Loss vs Step",
        out_path=out_dir / "loss.png",
        color="#2563eb",
    )
    plot_metric(
        train,
        metric="learning_rate",
        ylabel="Learning Rate",
        title="Learning Rate vs Step",
        out_path=out_dir / "learning_rate.png",
        color="#16a34a",
    )
    plot_metric(
        train,
        metric="grad_norm",
        ylabel="Grad Norm",
        title="Gradient Norm vs Step",
        out_path=out_dir / "grad_norm.png",
        color="#dc2626",
    )
    plot_metric(
        train,
        metric="epoch",
        ylabel="Epoch",
        title="Epoch vs Step",
        out_path=out_dir / "epoch.png",
        color="#9333ea",
    )
    plot_combined(train, out_dir / "combined.png")

    print(f"Saved {len(list(out_dir.glob('*.png')))} plots to {out_dir}")


if __name__ == "__main__":
    main()
