"""Gemma decoder tap + parallel Mimi speech-token head (Frisson E4B style)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from gemma_turkish.speech.codec import MimiSpeechCodec
from gemma_turkish.speech.config import SpeechTrainConfig

class LastLayerMix(nn.Module):
    """Learned softmax mix of the last N Gemma decoder hidden states."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        layers = hidden_states[-self.num_layers :]
        w = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(layers, dim=0)
        return (w.view(-1, 1, 1, 1) * stacked).sum(dim=0)


class ParallelSpeechHead(nn.Module):
    """
    Parallel Mimi head: summary of prompt → T frames × K codebooks.

    Uses learned per-frame biases (not a huge ``hidden × T×hidden`` projection) so
    E4B-scale ``head_hidden_dim`` stays near the Frisson ~152M param budget.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_codebooks: int,
        codebook_size: int,
        max_frames: int,
        head_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.num_codebooks = num_codebooks
        self.cond = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.GELU(),
        )
        self.frame_bias = nn.Parameter(torch.zeros(max_frames, head_hidden_dim))
        self.codebook_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_hidden_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Linear(head_hidden_dim, codebook_size),
                )
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, pooled: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Return logits (B, T, K, V)."""
        summary = self.cond(pooled)
        frame_h = summary.unsqueeze(1) + self.frame_bias[:num_frames].unsqueeze(0)
        logits = torch.stack([head(frame_h) for head in self.codebook_heads], dim=2)
        return logits


def _last_token_states(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, lengths]


def _mean_answer_pool(
    hidden: torch.Tensor,
    answer_start: int,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool mixed decoder states over generated answer tokens (B, H)."""
    answer_mask = attention_mask[:, answer_start:].float()
    answer_h = hidden[:, answer_start:, :]
    denom = answer_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (answer_h * answer_mask.unsqueeze(-1)).sum(dim=1) / denom


def speech_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over codebooks and frames; targets (B, K, T)."""
    b, t, k, v = logits.shape
    logits_flat = logits.permute(0, 2, 1, 3).reshape(b * k * t, v)
    targets_flat = targets.permute(0, 1, 2).reshape(b * k * t)
    mask_flat = frame_mask.unsqueeze(1).expand(b, k, t).reshape(b * k * t)
    if mask_flat.sum() == 0:
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    return (loss * mask_flat).sum() / mask_flat.sum()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(name.lower(), torch.bfloat16)


def _text_hidden_size(gemma: nn.Module) -> int:
    cfg = gemma.config
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        return int(text_cfg.hidden_size)
    return int(cfg.hidden_size)


def _is_gemma4_model_id(model_id: str) -> bool:
    mid = model_id.lower()
    return "gemma-4" in mid or "gemma4" in mid


def _load_gemma_backbone(model_id: str, dtype: torch.dtype, gradient_checkpointing: bool):
    """Load Gemma 4 multimodal LM or fall back to causal LM for older checkpoints."""
    if _is_gemma4_model_id(model_id):
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        gemma = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        if gradient_checkpointing:
            gemma.gradient_checkpointing_enable()
        return processor, gemma, True

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    gemma = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if gradient_checkpointing and hasattr(gemma, "gradient_checkpointing_enable"):
        gemma.gradient_checkpointing_enable()
    return tokenizer, gemma, False


def _tensor_to_list(value: Any) -> list[int]:
    if hasattr(value, "dim") and value.dim() == 2:
        value = value[0]
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _encoding_from_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    max_length: int,
) -> dict[str, Any]:
    """Normalize apply_chat_template output (BatchEncoding or token id list)."""
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors=None,
    )
    if hasattr(out, "get") and "input_ids" in out:
        input_ids = out["input_ids"]
        attention_mask = out.get("attention_mask")
    elif isinstance(out, dict):
        input_ids = out["input_ids"]
        attention_mask = out.get("attention_mask")
    else:
        input_ids = out
        attention_mask = None

    if isinstance(input_ids, str):
        return tokenizer(
            input_ids,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

    input_ids = _tensor_to_list(input_ids)
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    else:
        attention_mask = _tensor_to_list(attention_mask)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class GemmaSpeechModel(nn.Module):
    """
    Frozen Gemma + trainable layer-mix and parallel Mimi-token head.

    Tap: learned mix of the last ``num_tap_layers`` decoder states, before the LM head
    (Frisson / gemma4-audio pattern). Default backbone: ``google/gemma-4-E4B-it``.
    """

    def __init__(self, config: SpeechTrainConfig) -> None:
        super().__init__()
        self.config = config
        dtype = _resolve_dtype(config.gemma_dtype)

        self.processor, self.gemma, self._is_gemma4 = _load_gemma_backbone(
            config.gemma_model_id,
            dtype=dtype,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.codec = MimiSpeechCodec(
            model_id=config.mimi_model_id,
            num_codebooks=config.num_codebooks,
            sample_rate=config.mimi_sample_rate,
        )

        hidden_dim = _text_hidden_size(self.gemma)

        self.layer_mix = LastLayerMix(config.num_tap_layers)
        self.speech_head = ParallelSpeechHead(
            hidden_dim=hidden_dim,
            num_codebooks=config.num_codebooks,
            codebook_size=self.codec.codebook_size,
            max_frames=config.max_speech_frames,
            head_hidden_dim=config.head_hidden_dim,
        )

        if config.freeze_gemma:
            self.gemma.requires_grad_(False)
        if config.freeze_mimi:
            self.codec.freeze()

        self._generation_log_entries: list[dict[str, Any]] = []

    def _decode_generated_answers(
        self,
        generated_ids: torch.Tensor,
        prompt_len: int,
    ) -> list[str]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        answers: list[str] = []
        for row in generated_ids:
            ans_ids = row[prompt_len:]
            stop = ans_ids.shape[0]
            for j, tid in enumerate(ans_ids.tolist()):
                if tid in (pad_id, eos_id):
                    stop = j
                    break
            text = self.tokenizer.decode(ans_ids[:stop], skip_special_tokens=True)
            answers.append(text.strip())
        return answers

    def flush_generation_log(self, step: int, split: str) -> None:
        if not self.config.log_generated_outputs or not self._generation_log_entries:
            return
        path = Path(self.config.generated_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for entry in self._generation_log_entries:
                record = {"step": step, "split": split, **entry}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._generation_log_entries.clear()

    def reset_generation_log(self) -> None:
        if not self.config.log_generated_outputs:
            return
        path = Path(self.config.generated_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def format_prompt(self, text: str) -> str:
        return self.config.text_prompt_template.format(text=text.strip())

    def format_question(self, question: str) -> str:
        return self.config.generation_prompt_template.format(question=question.strip())

    def encode_question_prompt(self, question: str) -> dict[str, Any]:
        """User instruction for autoregressive answer generation (chat, gen prompt on)."""
        prompt = self.format_question(question)
        if self._is_gemma4:
            messages = [{"role": "user", "content": prompt}]
            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
            )
            input_ids = _tensor_to_list(encoded["input_ids"])
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = [1] * len(input_ids)
            else:
                attention_mask = _tensor_to_list(attention_mask)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return _encoding_from_chat_template(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                max_length=self.config.max_text_length,
            )

        return self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_text_length,
            padding=False,
            return_tensors=None,
        )

    def encode_text_prompt(self, text: str) -> dict[str, Any]:
        """Tokenize the Turkish speech-conditioning prompt for the text path."""
        prompt = self.format_prompt(text)
        if self._is_gemma4:
            messages = [{"role": "user", "content": prompt}]
            encoded = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
                return_dict=True,
            )
            input_ids = _tensor_to_list(encoded["input_ids"])
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = [1] * len(input_ids)
            else:
                attention_mask = _tensor_to_list(attention_mask)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return _encoding_from_chat_template(
                self.tokenizer,
                messages,
                add_generation_prompt=False,
                max_length=self.config.max_text_length,
            )

        return self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_text_length,
            padding=False,
            return_tensors=None,
        )

    def _gemma_forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.gemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Gemma forward did not return hidden_states; check output_hidden_states=True."
            )
        return hidden_states

    def _tokenize_questions_batch(
        self,
        questions: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_ids: list[list[int]] = []
        batch_mask: list[list[int]] = []
        for q in questions:
            enc = self.encode_question_prompt(q)
            batch_ids.append(enc["input_ids"])
            batch_mask.append(enc["attention_mask"])
        padded = self.tokenizer.pad(
            {"input_ids": batch_ids, "attention_mask": batch_mask},
            padding=True,
            return_tensors="pt",
        )
        return padded["input_ids"].to(device), padded["attention_mask"].to(device)

    def forward_generated_answer(
        self,
        questions: list[str],
        speech_token_ids: torch.Tensor,
        speech_frame_mask: torch.Tensor,
        answer_texts: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate a Turkish answer with frozen Gemma, then predict Mimi codes from
        mean-pooled hidden states over the generated answer span (Frisson blog v2).
        """
        device = speech_token_ids.device
        prompt_ids, prompt_mask = self._tokenize_questions_batch(questions, device)
        prompt_len = int(prompt_ids.shape[1])
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        gemma_was_training = self.gemma.training
        self.gemma.eval()
        with torch.no_grad():
            generated_ids = self.gemma.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        if gemma_was_training:
            self.gemma.train(False)

        if self.config.log_generated_outputs:
            decoded = self._decode_generated_answers(generated_ids, prompt_len)
            refs = answer_texts if answer_texts is not None else [None] * len(questions)
            for question, generated, reference in zip(questions, decoded, refs):
                self._generation_log_entries.append(
                    {
                        "question": question,
                        "generated": generated,
                        "reference": reference,
                    }
                )

        full_len = generated_ids.shape[1]
        full_mask = torch.zeros_like(generated_ids, dtype=torch.long)
        for i in range(generated_ids.size(0)):
            row = generated_ids[i]
            non_pad = (row != pad_id).nonzero(as_tuple=False)
            if non_pad.numel() > 0:
                last = int(non_pad[-1].item()) + 1
                full_mask[i, :last] = 1
            else:
                full_mask[i, :] = 1

        hidden_states = self._gemma_forward_hidden(generated_ids, full_mask)
        mixed = self.layer_mix(hidden_states)
        answer_start = min(prompt_len, full_len - 1)
        pooled = _mean_answer_pool(mixed, answer_start, full_mask)
        num_frames = speech_token_ids.shape[-1]
        logits = self.speech_head(pooled, num_frames=num_frames)
        loss = speech_token_loss(logits, speech_token_ids, speech_frame_mask)
        return {"loss": loss, "logits": logits}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        speech_token_ids: torch.Tensor | None = None,
        speech_frame_mask: torch.Tensor | None = None,
        questions: list[str] | None = None,
        answer_texts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if self.config.training_mode == "generated_answer":
            if questions is None or speech_token_ids is None or speech_frame_mask is None:
                raise ValueError(
                    "generated_answer mode requires questions, speech_token_ids, speech_frame_mask"
                )
            return self.forward_generated_answer(
                questions,
                speech_token_ids,
                speech_frame_mask,
                answer_texts=answer_texts,
            )

        if input_ids is None or attention_mask is None or speech_token_ids is None:
            raise ValueError("teacher_forced mode requires input_ids, attention_mask, speech tokens")
        hidden_states = self._gemma_forward_hidden(input_ids, attention_mask)
        mixed = self.layer_mix(hidden_states)
        pooled = _last_token_states(mixed, attention_mask)
        num_frames = speech_token_ids.shape[-1]
        logits = self.speech_head(pooled, num_frames=num_frames)
        loss = speech_token_loss(logits, speech_token_ids, speech_frame_mask)
        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def predict_speech_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_frames: int,
    ) -> torch.LongTensor:
        self.eval()
        hidden_states = self._gemma_forward_hidden(input_ids, attention_mask)
        mixed = self.layer_mix(hidden_states)
        pooled = _last_token_states(mixed, attention_mask)
        logits = self.speech_head(pooled, num_frames=num_frames)
        return logits.argmax(dim=-1).permute(0, 2, 1)  # (B, K, T)