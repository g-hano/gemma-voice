"""Batch collation with padding for variable-length speech token sequences."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class SpeechCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_speech_frames: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_frames = max(int(f["speech_token_ids"].shape[-1]) for f in features)
        num_cb = features[0]["speech_token_ids"].shape[0]
        batch = len(features)

        speech = torch.zeros(batch, num_cb, max_frames, dtype=torch.long)
        frame_mask = torch.zeros(batch, max_frames, dtype=torch.float32)

        for i, f in enumerate(features):
            codes = f["speech_token_ids"]
            t = codes.shape[-1]
            speech[i, :, :t] = codes
            frame_mask[i, :t] = 1.0

        out: dict[str, Any] = {
            "speech_token_ids": speech,
            "speech_frame_mask": frame_mask,
        }

        if "question" in features[0]:
            out["questions"] = [f["question"] for f in features]
            if "answer_text" in features[0]:
                out["answer_texts"] = [f["answer_text"] for f in features]
        else:
            text_batch = self.tokenizer.pad(
                {
                    "input_ids": [f["input_ids"] for f in features],
                    "attention_mask": [f["attention_mask"] for f in features],
                },
                padding=True,
                return_tensors="pt",
            )
            out["input_ids"] = text_batch["input_ids"]
            out["attention_mask"] = text_batch["attention_mask"]

        return out
