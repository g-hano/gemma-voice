"""Hugging Face Hub loading — one call loads Gemma + Mimi + speech head."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gemma_turkish.speech.config import SpeechTrainConfig
from gemma_turkish.speech.model import GemmaSpeechModel


def resolve_hub_dir(pretrained_model_name_or_path: str | Path) -> Path:
    """Local folder, speech_head.pt path, or HF repo id → directory with weights."""
    path = Path(pretrained_model_name_or_path)
    if path.is_file():
        return path.parent
    if path.is_dir() and (path / "speech_head.pt").is_file():
        return path.resolve()
    if path.is_dir() and (path / "config.json").is_file():
        pt = path / "speech_head.pt"
        if pt.is_file():
            return path.resolve()

    from huggingface_hub import snapshot_download

    repo_id = str(pretrained_model_name_or_path)
    return Path(snapshot_download(repo_id)).resolve()


def load_config_from_hub(repo_dir: Path) -> SpeechTrainConfig:
    pt = repo_dir / "speech_head.pt"
    state = torch.load(pt, map_location="cpu", weights_only=True)
    if "config" in state:
        cfg = SpeechTrainConfig.from_dict(state["config"])
    else:
        cfg = SpeechTrainConfig.from_dict(
            json.loads((repo_dir / "config.json").read_text(encoding="utf-8"))
        )
    cfg.gradient_checkpointing = False
    return cfg


class GemmaTurkishTTS(GemmaSpeechModel):
    """
    Turkish TTS: Gemma 4 E2B-it + Mimi + trained speech head in one object.

    Weights are **not** duplicated in the Hub repo (~520 MB adapters only).
    ``from_pretrained`` downloads this repo, then pulls the frozen backbone
    (``google/gemma-4-E2B-it``) and codec (``kyutai/mimi``) from their
    official checkpoints and attaches the trained ``layer_mix`` + ``speech_head``.

    Example::

        model = GemmaTurkishTTS.from_pretrained("Chan-Y/gemma-turkish-speech-e2b-mimi")
        model = model.to("cuda")
        wave = model.synthesize("Merhaba dünya.")
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device: str | torch.device | None = None,
        token: str | None = None,
        cache_dir: str | None = None,
        **_: Any,
    ) -> "GemmaTurkishTTS":
        """
        Load full TTS stack from Hub repo id or local export folder.

        ``pretrained_model_name_or_path`` can be:
        - ``"Chan-Y/gemma-turkish-speech-e2b-mimi"`` (HF Hub)
        - ``outputs/speech_head_v2/checkpoint-12000`` (local checkpoint dir)
        - path to ``speech_head.pt``
        """
        if cache_dir or token:
            from huggingface_hub import snapshot_download

            repo_dir = Path(
                snapshot_download(
                    str(pretrained_model_name_or_path),
                    cache_dir=cache_dir,
                    token=token,
                )
            ).resolve()
        else:
            repo_dir = resolve_hub_dir(pretrained_model_name_or_path)

        cfg = load_config_from_hub(repo_dir)
        model = cls(cfg)
        step = cls.load_trainable_checkpoint(model, repo_dir)
        model.eval()

        if device is not None:
            model = model.to(device)
        elif torch.cuda.is_available():
            model = model.cuda()

        model._hub_step = step  # noqa: SLF001
        model._hub_repo = str(pretrained_model_name_or_path)
        return model
