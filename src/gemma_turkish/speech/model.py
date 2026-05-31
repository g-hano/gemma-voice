"""Gemma decoder tap + parallel Mimi speech-token head (Frisson E4B style)."""
from __future__ import annotations
import hashlib
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


class AutoregressiveSpeechHead(nn.Module):
    """
    Transformer decoder over Mimi frames that cross-attends to the full sequence of
    Gemma hidden states (B, L, H), instead of a single pooled vector.

    - Temporal structure: causal self-attention over frames.
    - Text/content conditioning: cross-attention to every Gemma token state.
    - RVQ structure: within a frame, codebook ``k`` is predicted conditioned on the
      (teacher-forced) lower codebooks ``<k`` (residual quantization is sequential).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_codebooks: int,
        codebook_size: int,
        max_frames: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.max_frames = max_frames
        self.d_model = d_model

        self.memory_proj = nn.Linear(hidden_dim, d_model)
        self.code_embed = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks)]
        )
        self.bos = nn.Parameter(torch.zeros(d_model))
        self.frame_pos = nn.Parameter(torch.zeros(max_frames, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.cb_heads = nn.ModuleList(
            [nn.Linear(d_model, codebook_size) for _ in range(num_codebooks)]
        )
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.bos, std=0.02)

    def _causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1
        )

    def _frame_inputs(self, codes: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decoder inputs: frame t sees the *previous* frame's codes."""
        b, k, t = codes.shape
        emb = sum(self.code_embed[i](codes[:, i, :]) for i in range(k))  # (B, T, d)
        dec_in = torch.zeros(b, t, self.d_model, dtype=emb.dtype, device=emb.device)
        dec_in[:, 0] = self.bos.to(emb.dtype)
        if t > 1:
            dec_in[:, 1:] = emb[:, :-1]
        dec_in = dec_in + self.frame_pos[:t].unsqueeze(0).to(emb.dtype)
        return dec_in

    def _run_decoder(
        self,
        dec_in: torch.Tensor,
        mem: torch.Tensor,
        mem_pad: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        t = dec_in.size(1)
        tgt_pad = frame_mask == 0
        return self.decoder(
            tgt=dec_in,
            memory=mem,
            tgt_mask=self._causal_mask(t, dec_in.device),
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )

    def _codebook_logits(self, h: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        h = self.out_norm(h)
        cond = h
        logits = []
        for k in range(self.num_codebooks):
            logits.append(self.cb_heads[k](cond))
            if k < self.num_codebooks - 1:
                cond = cond + self.code_embed[k](codes[:, k, :])
        return torch.stack(logits, dim=2)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        codes: torch.Tensor,
        frame_mask: torch.Tensor,
        sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        """Teacher-forced logits (B, T, K, V). ``codes`` is (B, K, T)."""
        mem = self.memory_proj(memory)
        mem_pad = memory_mask == 0
        b, _k, t = codes.shape
        cond_codes = codes

        if sampling_prob > 0 and self.training and t > 1:
            dec_in_tf = self._frame_inputs(codes)
            h_tf = self._run_decoder(dec_in_tf, mem, mem_pad, frame_mask)
            pred = self._codebook_logits(h_tf, codes).argmax(dim=-1).permute(0, 2, 1)
            use_pred = torch.rand(b, t - 1, 1, device=codes.device) < sampling_prob
            use_pred = use_pred.transpose(1, 2).expand(-1, _k, -1)
            cond_codes = codes.clone()
            cond_codes[:, :, : t - 1] = torch.where(
                use_pred, pred[:, :, : t - 1], codes[:, :, : t - 1]
            )

        dec_in = self._frame_inputs(cond_codes)
        h = self._run_decoder(dec_in, mem, mem_pad, frame_mask)
        return self._codebook_logits(h, codes)

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.LongTensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        scaled = logits / max(temperature, 1e-5)
        probs = F.softmax(scaled, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = cum <= top_p
            keep[..., 0] = True
            filtered = sorted_probs * keep
            filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            pick = torch.multinomial(filtered, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def generate(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        num_frames: int,
        *,
        early_stop: bool = True,
        min_frames: int = 20,
        patience: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> torch.LongTensor:
        """Autoregressively sample codes (B, K, T) frame-by-frame, codebook-by-codebook."""
        mem = self.memory_proj(memory)
        mem_pad = memory_mask == 0
        b = memory.size(0)
        device = memory.device
        num_frames = min(num_frames, self.max_frames)
        inputs_emb: list[torch.Tensor] = []
        generated: list[torch.Tensor] = []
        prev_codes: torch.Tensor | None = None
        stable_run = 0
        prev_cb0: torch.Tensor | None = None
        for t in range(num_frames):
            if t == 0:
                step_in = self.bos.to(mem.dtype).unsqueeze(0).expand(b, -1)
            else:
                step_in = sum(
                    self.code_embed[i](prev_codes[:, i]) for i in range(self.num_codebooks)
                )
            inputs_emb.append(step_in)
            seq = torch.stack(inputs_emb, dim=1) + self.frame_pos[: t + 1].unsqueeze(0).to(mem.dtype)
            h = self.decoder(
                tgt=seq,
                memory=mem,
                tgt_mask=self._causal_mask(t + 1, device),
                memory_key_padding_mask=mem_pad,
            )
            h_t = self.out_norm(h[:, -1])
            cond = h_t
            frame_codes = []
            for k in range(self.num_codebooks):
                tok = self._sample_from_logits(self.cb_heads[k](cond), temperature, top_p)
                frame_codes.append(tok)
                cond = cond + self.code_embed[k](tok)
            prev_codes = torch.stack(frame_codes, dim=1)  # (B, K)
            generated.append(prev_codes)
            if early_stop:
                cb0 = frame_codes[0]
                if prev_cb0 is not None and torch.equal(cb0, prev_cb0):
                    stable_run += 1
                else:
                    stable_run = 0
                prev_cb0 = cb0
                if t + 1 >= min_frames and stable_run >= patience:
                    break
        return torch.stack(generated, dim=2).long()  # (B, K, T)


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
    codebook_weights: list[float] | None = None,
) -> torch.Tensor:
    """Cross-entropy over codebooks and frames; targets (B, K, T)."""
    b, t, k, v = logits.shape
    if codebook_weights is None:
        codebook_weights = [2.0 ** (k - 1 - i) for i in range(k)]
    total_w = sum(codebook_weights)
    loss_sum = logits.new_zeros(())
    for ki in range(k):
        logits_k = logits[:, :, ki, :].reshape(b * t, v)
        targets_k = targets[:, ki, :].reshape(b * t)
        mask_k = frame_mask.reshape(b * t)
        if mask_k.sum() == 0:
            continue
        ce = F.cross_entropy(logits_k, targets_k, reduction="none")
        loss_sum = loss_sum + codebook_weights[ki] * (ce * mask_k).sum() / mask_k.sum()
    return loss_sum / max(total_w, 1e-8)


def estimate_speech_frames(text: str, config: SpeechTrainConfig) -> int:
    """Heuristic frame count from text length (12.5 Hz Mimi frames)."""
    cps = max(config.synth_chars_per_second, 1.0)
    duration_sec = len(text.strip()) / cps
    frames = int(duration_sec * 12.5) + config.eval_audio_frame_margin
    return max(25, min(frames, config.eval_audio_max_frames))


def trim_trailing_silence(
    wave: torch.Tensor,
    sample_rate: int,
    *,
    threshold: float = 0.015,
    window_ms: int = 40,
    pad_ms: int = 80,
) -> torch.Tensor:
    """Drop low-energy tail after speech (fixes long post-speech silence in eval WAVs)."""
    import numpy as np

    x = wave.squeeze().detach().float().cpu().numpy()
    if x.size == 0:
        return wave.squeeze()
    win = max(1, int(sample_rate * window_ms / 1000))
    pad = int(sample_rate * pad_ms / 1000)
    n_win = max(1, (x.size + win - 1) // win)
    padded = np.pad(x, (0, n_win * win - x.size))
    rms = np.sqrt(
        np.maximum(
            np.mean(padded.reshape(n_win, win) ** 2, axis=1),
            1e-12,
        )
    )
    active = np.where(rms > threshold)[0]
    if active.size == 0:
        return wave.squeeze()
    end = min(x.size, (int(active[-1]) + 1) * win + pad)
    return torch.from_numpy(x[:end])


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


def _device_map_kwargs(
    device_map: str | None, max_gpu_memory_gib: float | None
) -> dict[str, Any]:
    """Build from_pretrained kwargs for optional accelerate device_map / CPU offload."""
    if not device_map:
        return {}
    kwargs: dict[str, Any] = {"device_map": device_map}
    if max_gpu_memory_gib is not None and torch.cuda.is_available():
        kwargs["max_memory"] = {0: f"{max_gpu_memory_gib}GiB", "cpu": "64GiB"}
    return kwargs


def _load_gemma_backbone(
    model_id: str,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
    device_map: str | None = None,
    max_gpu_memory_gib: float | None = None,
):
    """Load Gemma 4 multimodal LM or fall back to causal LM for older checkpoints."""
    offload = _device_map_kwargs(device_map, max_gpu_memory_gib)
    if _is_gemma4_model_id(model_id):
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        gemma = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            **offload,
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
        **offload,
    )
    if gradient_checkpointing and hasattr(gemma, "gradient_checkpointing_enable"):
        gemma.gradient_checkpointing_enable()
    return tokenizer, gemma, False


def _flatten_token_list(value: Any) -> list[int]:
    """Processor/tokenizer output → 1D ``list[int]`` (drops a leading batch dim if present)."""
    while True:
        if value is None:
            return []
        if hasattr(value, "dim"):
            if value.dim() == 0:
                return [int(value.item())]
            if value.dim() >= 2:
                value = value[0]
                continue
            return [int(x) for x in value.tolist()]
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            if isinstance(value[0], (list, tuple)):
                value = value[0]
                continue
            return [int(x) for x in value]
        return [int(value)]


def _ensure_batched_2d(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"`{name}` must be 1D or 2D, got shape {tuple(t.shape)}")
    return t


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

    input_ids = _flatten_token_list(input_ids)
    if attention_mask is None:
        attention_mask = [1] * len(input_ids)
    else:
        attention_mask = _flatten_token_list(attention_mask)
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
            device_map=config.gemma_device_map,
            max_gpu_memory_gib=config.gemma_max_gpu_memory_gib,
        )
        self._gemma_offloaded = bool(config.gemma_device_map)
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
        self.head_type = getattr(config, "head_type", "autoregressive")
        if self.head_type == "autoregressive":
            self.speech_head = AutoregressiveSpeechHead(
                hidden_dim=hidden_dim,
                num_codebooks=config.num_codebooks,
                codebook_size=self.codec.codebook_size,
                max_frames=config.max_speech_frames,
                d_model=config.speech_decoder_d_model,
                num_layers=config.speech_decoder_layers,
                num_heads=config.speech_decoder_heads,
                ffn_dim=config.speech_decoder_ffn_dim,
                dropout=config.speech_decoder_dropout,
            )
        else:
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
        self._global_step: int = 0

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def scheduled_sampling_prob(self) -> float:
        cfg = self.config
        if not cfg.scheduled_sampling_enabled or self.head_type != "autoregressive":
            return 0.0
        ramp = max(1, cfg.scheduled_sampling_ramp_steps)
        t = min(1.0, self._global_step / ramp)
        return t * cfg.scheduled_sampling_max_prob

    def _speech_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        return speech_token_loss(
            logits,
            targets,
            frame_mask,
            codebook_weights=self.config.codebook_loss_weights,
        )

    @classmethod
    def load_trainable_checkpoint(cls, model: "GemmaSpeechModel", path: str | Path) -> int:
        """Load layer_mix + speech_head weights; returns global_step if trainer_state exists."""
        ckpt_dir = Path(path)
        pt = ckpt_dir / "speech_head.pt" if ckpt_dir.is_dir() else ckpt_dir
        if not pt.is_file():
            raise FileNotFoundError(f"No speech_head.pt at {pt}")
        state = torch.load(pt, map_location="cpu", weights_only=True)
        model.layer_mix.load_state_dict(state["layer_mix"])
        model.speech_head.load_state_dict(state["speech_head"])
        trainer_state = ckpt_dir / "trainer_state.json" if ckpt_dir.is_dir() else None
        step = 0
        if trainer_state is not None and trainer_state.is_file():
            step = int(json.loads(trainer_state.read_text(encoding="utf-8")).get("global_step", 0))
        return step

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
            input_ids = _flatten_token_list(encoded["input_ids"])
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = [1] * len(input_ids)
            else:
                attention_mask = _flatten_token_list(attention_mask)
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
            input_ids = _flatten_token_list(encoded["input_ids"])
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = [1] * len(input_ids)
            else:
                attention_mask = _flatten_token_list(attention_mask)
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

    def _feature_cache_path(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Path | None:
        if not getattr(self.config, "cache_gemma_features", False):
            return None
        if self._gemma_offloaded or input_ids.size(0) != 1:
            return None  # only safe / useful for single, un-padded sequences
        real = input_ids[0][attention_mask[0].bool()].tolist()
        key = f"{self.config.num_tap_layers}_" + ",".join(map(str, real))
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        root = Path(self.config.gemma_feature_cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{digest}.pt"

    def _gemma_forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache_path = self._feature_cache_path(input_ids, attention_mask)
        if cache_path is not None and cache_path.is_file():
            device = next(self.layer_mix.parameters()).device
            dtype = _resolve_dtype(self.config.gemma_dtype)
            cached = torch.load(cache_path, weights_only=True)
            return tuple(t.to(device=device, dtype=dtype) for t in cached["last_states"])

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
        if cache_path is not None:
            n = self.config.num_tap_layers
            last = [h.detach().to(torch.float16).cpu() for h in hidden_states[-n:]]
            torch.save({"last_states": last}, cache_path)
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
        input_ids = _ensure_batched_2d(padded["input_ids"], "input_ids").to(device)
        attention_mask = _ensure_batched_2d(padded["attention_mask"], "attention_mask").to(device)
        return input_ids, attention_mask

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
        if self.head_type == "autoregressive":
            # Cross-attend only over the generated answer span.
            answer_start = min(prompt_len, full_len - 1)
            mem = mixed[:, answer_start:, :]
            mem_mask = full_mask[:, answer_start:]
            logits = self.speech_head(
                mem, mem_mask, speech_token_ids, speech_frame_mask,
                sampling_prob=self.scheduled_sampling_prob(),
            )
        else:
            answer_start = min(prompt_len, full_len - 1)
            pooled = _mean_answer_pool(mixed, answer_start, full_mask)
            logits = self.speech_head(pooled, num_frames=speech_token_ids.shape[-1])
        loss = self._speech_loss(logits, speech_token_ids, speech_frame_mask)
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
        if speech_frame_mask is None:
            speech_frame_mask = torch.ones(
                speech_token_ids.shape[0], speech_token_ids.shape[-1],
                dtype=torch.float32, device=speech_token_ids.device,
            )
        hidden_states = self._gemma_forward_hidden(input_ids, attention_mask)
        mixed = self.layer_mix(hidden_states)
        if self.head_type == "autoregressive":
            logits = self.speech_head(
                mixed,
                attention_mask,
                speech_token_ids,
                speech_frame_mask,
                sampling_prob=self.scheduled_sampling_prob(),
            )
        else:
            pooled = _last_token_states(mixed, attention_mask)
            logits = self.speech_head(pooled, num_frames=speech_token_ids.shape[-1])
        loss = self._speech_loss(logits, speech_token_ids, speech_frame_mask)
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
        if self.head_type == "autoregressive":
            return self.speech_head.generate(
                mixed,
                attention_mask,
                num_frames,
                early_stop=self.config.synth_early_stop,
                min_frames=self.config.synth_min_frames,
                patience=self.config.synth_early_stop_patience,
                temperature=self.config.synth_temperature,
                top_p=self.config.synth_top_p,
            )
        pooled = _last_token_states(mixed, attention_mask)
        logits = self.speech_head(pooled, num_frames=num_frames)
        return logits.argmax(dim=-1).permute(0, 2, 1)  # (B, K, T)

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        num_frames: int | None = None,
        *,
        trim_silence: bool | None = None,
    ) -> torch.Tensor:
        """Text → predicted Mimi codes → waveform (1, samples). Conditions on the transcript."""
        self.eval()
        device = next(self.speech_head.parameters()).device
        enc = self.encode_text_prompt(text)
        padded = self.tokenizer.pad(
            {"input_ids": [enc["input_ids"]], "attention_mask": [enc["attention_mask"]]},
            padding=True,
            return_tensors="pt",
        )
        input_ids = _ensure_batched_2d(padded["input_ids"], "input_ids").to(device)
        attention_mask = _ensure_batched_2d(padded["attention_mask"], "attention_mask").to(device)
        n = num_frames or estimate_speech_frames(text, self.config)
        codes = self.predict_speech_tokens(input_ids, attention_mask, n)
        codes = codes.to(next(self.codec.parameters()).device)
        waveform = self.codec.decode_codes(codes)  # (B, 1, samples)
        wave = waveform[0].detach().float().cpu()
        do_trim = (
            self.config.eval_audio_trim_silence if trim_silence is None else trim_silence
        )
        if do_trim:
            wave = trim_trailing_silence(wave, self.config.mimi_sample_rate)
        return wave