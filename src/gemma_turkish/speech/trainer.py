"""Hugging Face Trainer wiring for the speech head."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any
import torch
from transformers import TrainerCallback
from gemma_turkish.speech.collator import SpeechCollator
from gemma_turkish.speech.config import SpeechTrainConfig
from gemma_turkish.speech.data import TurkishSpeechDataset, load_turkish_speech_dataset, train_val_split

if TYPE_CHECKING:
    from gemma_turkish.speech.model import GemmaSpeechModel


class EvalAudioCallback(TrainerCallback):
    """After each eval, synthesize a few fixed eval texts to WAV for listening."""

    def __init__(self, config: SpeechTrainConfig, samples: list[dict[str, Any]]) -> None:
        self.config = config
        self.samples = samples

    def on_evaluate(self, args, state, control, **kwargs) -> None:
        model = kwargs.get("model")
        if model is None or not self.samples:
            return
        import soundfile as sf

        out_dir = Path(self.config.eval_audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        step = int(state.global_step)
        sr = int(self.config.mimi_sample_rate)
        was_training = model.training
        model.eval()
        try:
            for i, sample in enumerate(self.samples[: self.config.eval_audio_samples]):
                text = str(sample["text"])
                ref_frames = sample.get("num_frames")
                if ref_frames is not None:
                    n = min(
                        int(ref_frames) + self.config.eval_audio_frame_margin,
                        self.config.eval_audio_max_frames,
                    )
                else:
                    n = None
                try:
                    wave = model.synthesize(text, num_frames=n)
                    audio = wave.squeeze().numpy()
                    dur = len(audio) / sr
                    path = out_dir / f"step{step:06d}_sample{i}.wav"
                    sf.write(str(path), audio, sr)
                    meta = f"{text}\n\nframes={n or 'auto'} duration={dur:.2f}s"
                    (out_dir / f"step{step:06d}_sample{i}.txt").write_text(
                        meta, encoding="utf-8"
                    )
                    print(f"[eval-audio] step {step} sample {i}: {dur:.2f}s ({n or 'auto'} frames)")
                except Exception as exc:  # keep training alive on synth errors
                    print(f"[eval-audio] sample {i} failed: {exc}")
        finally:
            if was_training:
                model.train()

def build_trainer(
    model: GemmaSpeechModel,
    config: SpeechTrainConfig,
    *,
    smoke: bool = False,
) -> Any:
    """Build a HF ``Trainer`` (imported lazily for faster package import checks)."""
    from transformers import Trainer, TrainingArguments
    from gemma_turkish.speech.model import GemmaSpeechModel as _GemmaSpeechModel
    class SpeechHeadTrainer(Trainer):
        def on_train_begin(self, args, state, control, **kwargs) -> None:
            speech_model: _GemmaSpeechModel = self.model  # type: ignore[assignment]
            if speech_model.config.log_generated_outputs:
                speech_model.reset_generation_log()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
            speech_model: _GemmaSpeechModel = model  # type: ignore[assignment]
            if speech_model.config.log_generated_outputs:
                split = "train" if model.training else "eval"
                step = int(self.state.global_step) if self.state is not None else 0
                speech_model.flush_generation_log(step, split)
            return loss

        def _save_speech_head(self, output_dir: str) -> None:
            """Save trainable adapters only (frozen Gemma/Mimi stay on HF hub)."""
            speech_model: _GemmaSpeechModel = self.model  # type: ignore[assignment]
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "layer_mix": speech_model.layer_mix.state_dict(),
                    "speech_head": speech_model.speech_head.state_dict(),
                    "config": speech_model.config.to_dict(),
                },
                Path(output_dir) / "speech_head.pt",
            )

        def _save(self, output_dir: str | None = None, state_dict=None) -> None:
            if output_dir is None:
                return
            self._save_speech_head(output_dir)

        def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
            if output_dir is None:
                output_dir = self.args.output_dir
            self._save_speech_head(output_dir)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if model is None:
                model = self.model
            ckpt = Path(resume_from_checkpoint)
            pt = ckpt / "speech_head.pt"
            if pt.is_file():
                step = _GemmaSpeechModel.load_trainable_checkpoint(model, ckpt)
                print(f"Loaded speech_head weights from {ckpt} (was at step {step})")
                # HF Trainer expects pytorch_model.bin; we only save speech_head.pt.
                return
            return super()._load_from_checkpoint(resume_from_checkpoint, model)

    full = load_turkish_speech_dataset(config)
    train_ds, eval_ds = train_val_split(full, config.val_fraction, config.seed)

    if config.max_eval_samples is not None and len(eval_ds) > config.max_eval_samples:
        eval_ds = eval_ds.select(range(config.max_eval_samples))

    encode_fn = (
        None
        if config.training_mode == "generated_answer"
        else model.encode_text_prompt
    )
    train_set = TurkishSpeechDataset(
        train_ds, encode_fn, model.codec, config, split_label="train"
    )
    eval_set = TurkishSpeechDataset(
        eval_ds, encode_fn, model.codec, config, split_label="eval"
    )

    eval_audio_samples = []
    for i in range(min(config.eval_audio_samples, len(eval_set))):
        item = eval_set[i]
        eval_audio_samples.append(
            {"text": item["text"], "num_frames": int(item["num_frames"])}
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    eval_strategy = "no" if smoke else "steps"
    save_strategy = "no" if smoke else "steps"
    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps if not smoke else None,
        save_strategy=save_strategy,
        save_steps=config.save_steps if not smoke else None,
        save_total_limit=3,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to=config.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        label_names=[],
        seed=config.seed,
    )

    callbacks = []
    if not smoke and config.eval_audio_samples > 0 and eval_audio_samples:
        callbacks.append(EvalAudioCallback(config, eval_audio_samples))

    return SpeechHeadTrainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=SpeechCollator(model.tokenizer),
        callbacks=callbacks,
        optimizers=(
            torch.optim.AdamW(
                trainable, lr=config.learning_rate, weight_decay=config.weight_decay
            ),
            None,
        ),
    )