"""Training configuration for Gemma → Mimi speech-head experiments."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass
class SpeechTrainConfig:
    """Hyperparameters aligned with Frisson Labs' Gemma4 E4B smoke test, Turkish defaults."""
    # --- Gemma backbone (Gemma 4 E4B-it multimodal) ---
    gemma_model_id: str = "google/gemma-4-E4B-it"
    freeze_gemma: bool = True
    gemma_dtype: str = "bfloat16"  # float16 | bfloat16 | float32
    num_tap_layers: int = 6  # E4B: 42 layers → tap last 6 (blog / gemma4-audio)
    gradient_checkpointing: bool = True

    # --- Mimi codec (blog: 8 codebooks @ 12.5 Hz, 24 kHz) ---
    mimi_model_id: str = "kyutai/mimi"
    freeze_mimi: bool = True
    num_codebooks: int = 8
    mimi_sample_rate: int = 24000
    max_audio_seconds: float = 30.0
    min_audio_seconds: float = 0.5

    # --- Speech head ---
    # "autoregressive": cross-attention Transformer decoder over Mimi frames (recommended).
    # "parallel": legacy pooled-vector head (cannot model temporal/content structure).
    head_type: str = "autoregressive"
    head_hidden_dim: int = 2560  # parallel head only
    max_speech_frames: int = 375  # 30 s * 12.5 Hz

    # --- Autoregressive speech decoder (cross-attends to full Gemma hidden states) ---
    speech_decoder_d_model: int = 1024
    speech_decoder_layers: int = 6
    speech_decoder_heads: int = 8
    speech_decoder_ffn_dim: int = 4096
    speech_decoder_dropout: float = 0.1

    # --- Turkish text conditioning (blog EN: Say this naturally as speech:\n{text}) ---
    text_prompt_template: str = "Bunu doğal bir konuşma gibi söyle:\n{text}"
    max_text_length: int = 512
    min_text_chars: int = 4
    max_text_chars: int = 500

    # teacher_forced: repeat phrase (legacy). generated_answer: Gemma generate → answer hiddens → Mimi.
    training_mode: str = "generated_answer"
    max_new_tokens: int = 48
    generation_prompt_template: str = "Şunu kısa ve Türkçe yanıtla:\n{question}"
    # Synthetic question from transcript when dataset has no question column (teacher WAV = reference).
    generated_question_template: str = (
        "Aşağıdaki konuda kısa bir Türkçe cümle söyle:\n{context}"
    )
    dataset_question_column: str | None = None
    # Deprecated alias; maps to training_mode in load_config().
    use_generated_answer_states: bool = False

    # --- Data (default: CC BY 4.0 synthetic Turkish TTS on HF) ---
    dataset_name: str = "Anilosan15/Synthetic_Turkish_TTS_Data"
    dataset_config: str | None = None  # omit for single-config datasets (e.g. default)
    dataset_split: str = "train"
    dataset_text_column: str = "text"
    dataset_audio_column: str = "audio"
    max_samples: int | None = None  # None = use full dataset split
    val_fraction: float = 0.1
    max_eval_samples: int | None = 200  # cap eval set so eval stays cheap
    # Quality gate (text length, decodable audio, duration). Off by default for curated HF TTS sets.
    filter_dataset: bool = False
    use_demo_dataset: bool = False
    cache_speech_tokens: bool = True
    cache_dir: str = "data/speech_token_cache"
    # Cache frozen Gemma last-N hidden states to disk (skips the Gemma forward each step).
    # Big speedup but uses disk (~few MB/sample in fp16); the trainable layer-mix is applied
    # on the fly so caching the raw frozen states stays correct.
    cache_gemma_features: bool = False
    gemma_feature_cache_dir: str = "data/gemma_feature_cache"

    # --- Gemma placement / offload (16 GB GPUs) ---
    # None = whole backbone on the training device. "auto"/"balanced" = accelerate device_map
    # with CPU offload (frozen forward only, no optimizer state, so offload is safe but slower).
    gemma_device_map: str | None = None
    gemma_max_gpu_memory_gib: float | None = None  # e.g. 12.0 to reserve VRAM for the head

    # --- Optimization ---
    output_dir: str = "outputs/speech_head"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 4500
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 0

    # --- Eval audio synthesis (decode a sample to WAV after every eval) ---
    eval_audio_samples: int = 1  # how many eval texts to synthesize each eval
    eval_audio_max_frames: int = 150  # cap generated frames (12.5 Hz → ~12 s) to keep it fast
    eval_audio_dir: str | None = None  # default: {output_dir}/eval_audio

    # --- Extra ---
    report_to: str = "none"  # none | tensorboard | wandb
    log_generated_outputs: bool = True
    generated_log_path: str | None = None  # default: {output_dir}/generated_outputs.jsonl

    def resolve_paths(self, project_root: Path | None = None) -> None:
        root = project_root or Path.cwd()
        self.cache_dir = str((root / self.cache_dir).resolve())
        self.gemma_feature_cache_dir = str((root / self.gemma_feature_cache_dir).resolve())
        self.output_dir = str((root / self.output_dir).resolve())
        if self.eval_audio_dir is None:
            self.eval_audio_dir = str((Path(self.output_dir) / "eval_audio").resolve())
        else:
            ea = Path(self.eval_audio_dir)
            self.eval_audio_dir = str((ea if ea.is_absolute() else root / ea).resolve())
        if self.generated_log_path is None:
            self.generated_log_path = str((Path(self.output_dir) / "generated_outputs.jsonl").resolve())
        else:
            log_p = Path(self.generated_log_path)
            if not log_p.is_absolute():
                self.generated_log_path = str((root / log_p).resolve())
            else:
                self.generated_log_path = str(log_p.resolve())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeechTrainConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpeechTrainConfig:
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

def _apply_training_mode_compat(data: dict[str, Any]) -> dict[str, Any]:
    """Map legacy ``use_generated_answer_states`` to ``training_mode``."""
    out = dict(data)
    if out.get("use_generated_answer_states") and out.get("training_mode") in (
        None,
        "teacher_forced",
    ):
        out["training_mode"] = "generated_answer"
    return out


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> SpeechTrainConfig:
    root = project_root or Path.cwd()
    if path:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        data = _apply_training_mode_compat(raw)
        cfg = SpeechTrainConfig.from_dict(data)
    else:
        cfg = SpeechTrainConfig()
    if overrides:
        merged = _apply_training_mode_compat(
            {**cfg.to_dict(), **{k: v for k, v in overrides.items() if v is not None}}
        )
        cfg = SpeechTrainConfig.from_dict(merged)
    cfg.resolve_paths(root)
    return cfg