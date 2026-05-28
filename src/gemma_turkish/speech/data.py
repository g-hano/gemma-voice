"""Turkish speech–text datasets and Mimi token caching."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from datasets import Dataset

from gemma_turkish.speech.codec import MimiSpeechCodec
from gemma_turkish.speech.config import SpeechTrainConfig

# Turkish demo QA (generated_answer smoke / offline)
_DEMO_QA: list[tuple[str, str]] = [
    ("Ses kodu denemesi nedir?", "Merhaba, bu bir ses kodu denemesidir."),
    (
        "Donmuş katmanlar üzerinden konuşma nasıl koşullanır?",
        "Küçük adaptör donmuş katmanlar üzerinden net konuşmayı koşullar.",
    ),
    ("Eğitim ne yapar?", "Eğitim kısa Türkçe cümleleri doğal konuşmaya eşler."),
    (
        "Konuşma başlığı ne işe yarar?",
        "Konuşma başlığı metin istemlerinden ses üretir.",
    ),
]

def _resample_mono(audio: dict[str, Any], target_sr: int) -> np.ndarray:
    array = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    if sr == target_sr:
        return array
    import librosa
    return librosa.resample(array, orig_sr=sr, target_sr=target_sr)

def _truncate_audio(wave: np.ndarray, sample_rate: int, max_seconds: float) -> np.ndarray:
    max_samples = int(sample_rate * max_seconds)
    if wave.shape[0] > max_samples:
        return wave[:max_samples]
    return wave

def _normalize_audio_dict(
    audio: Any,
    audio_col: str,
) -> dict[str, Any] | None:
    """Return {array, sampling_rate} or None if unusable."""
    if audio is None:
        return None
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            return audio
        if audio.get("bytes"):
            import io
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(audio["bytes"]))
            wave = np.asarray(data, dtype=np.float32)
            if wave.ndim > 1:
                wave = wave.mean(axis=-1)
            return {"array": wave, "sampling_rate": int(sr)}

        path = audio.get("path")
        if path:
            import soundfile as sf
            data, sr = sf.read(path)
            wave = np.asarray(data, dtype=np.float32)
            if wave.ndim > 1:
                wave = wave.mean(axis=-1)
            return {"array": wave, "sampling_rate": int(sr)}
    return None

def _row_passes_filters(row: dict[str, Any], config: SpeechTrainConfig) -> bool:
    text_col = config.dataset_text_column
    audio_col = config.dataset_audio_column
    text = str(row.get(text_col) or "").strip()
    if len(text) < config.min_text_chars or len(text) > config.max_text_chars:
        return False
    audio = _normalize_audio_dict(row.get(audio_col), audio_col)
    if audio is None:
        return False
    arr = np.asarray(audio["array"], dtype=np.float32)
    if arr.size == 0:
        return False
    sr = int(audio["sampling_rate"])
    duration = arr.shape[0] / max(sr, 1)
    return config.min_audio_seconds <= duration <= config.max_audio_seconds

def _question_for_row(row: dict[str, Any], config: SpeechTrainConfig) -> str:
    qcol = config.dataset_question_column
    if qcol and row.get(qcol):
        return str(row[qcol]).strip()
    text = str(row.get(config.dataset_text_column) or "").strip()
    return config.generated_question_template.format(context=text)


class TurkishSpeechDataset(TorchDataset):
    """Audio + Turkish text → Mimi teacher codes; optional Gemma repeat-prompt tokens."""
    def __init__(
        self,
        hf_dataset: "Dataset",
        encode_prompt: Callable[[str], dict[str, Any]] | None,
        codec: MimiSpeechCodec,
        config: SpeechTrainConfig,
        split_label: str = "train",

    ) -> None:
        self.hf_dataset = hf_dataset
        self.encode_prompt = encode_prompt
        self.codec = codec
        self.config = config
        self.split_label = split_label
        self._generated_mode = config.training_mode == "generated_answer"
        self.cache_root = Path(config.cache_dir) / split_label
        if config.cache_speech_tokens:
            self.cache_root.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _cache_path(self, index: int, text: str) -> Path:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return self.cache_root / f"{index:06d}_{digest}.pt"

    def _encode_audio(self, wave: np.ndarray, index: int, text: str) -> torch.Tensor:
        cache_path = self._cache_path(index, text)
        if self.config.cache_speech_tokens and cache_path.is_file():
            cached = torch.load(cache_path, weights_only=True)
            return cached["codes"]

        tensor = torch.from_numpy(wave).float()
        encoded = self.codec.encode_waveform(tensor, sample_rate=self.config.mimi_sample_rate)
        codes = encoded.audio_codes.squeeze(0).cpu()
        if self.config.cache_speech_tokens:
            torch.save({"codes": codes}, cache_path)
        return codes

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.hf_dataset[index]
        text = str(row[self.config.dataset_text_column]).strip()

        audio = _normalize_audio_dict(row[self.config.dataset_audio_column], self.config.dataset_audio_column)
        if audio is None:
            raise ValueError(f"Row {index} has no decodable audio")

        wave = _resample_mono(audio, self.config.mimi_sample_rate)
        wave = _truncate_audio(wave, self.config.mimi_sample_rate, self.config.max_audio_seconds)

        codes = self._encode_audio(wave, index, text)
        num_frames = int(codes.shape[-1])
        frame_mask = torch.ones(num_frames, dtype=torch.float32)
        out: dict[str, Any] = {
            "speech_token_ids": codes.long(),
            "speech_frame_mask": frame_mask,
            "text": text,
            "num_frames": num_frames,
        }
        if self._generated_mode:
            out["question"] = _question_for_row(row, self.config)
            out["answer_text"] = text
        else:
            if self.encode_prompt is None:
                raise RuntimeError("teacher_forced mode requires encode_prompt")
            tokenized = self.encode_prompt(text)
            out["input_ids"] = tokenized["input_ids"]
            out["attention_mask"] = tokenized["attention_mask"]
        return out


def build_demo_dataset(config: SpeechTrainConfig) -> "Dataset":
    """Tiny in-memory set: synthetic noise WAV + Turkish QA or repeat sentences."""
    from datasets import Dataset

    rng = np.random.default_rng(config.seed)
    sr = config.mimi_sample_rate
    duration = 2.0
    samples = int(sr * duration)
    rows = []
    qcol = config.dataset_question_column or "question"
    if config.training_mode == "generated_answer":
        pairs = _DEMO_QA
    else:
        pairs = [(a, a) for _, a in _DEMO_QA]
    for question, sentence in pairs:
        wave = rng.standard_normal(samples).astype(np.float32) * 0.05
        row = {
            config.dataset_text_column: sentence,
            config.dataset_audio_column: {"array": wave, "sampling_rate": sr},
        }
        if config.training_mode == "generated_answer":
            row[qcol] = question
        rows.append(row)
    return Dataset.from_list(rows)

def load_turkish_speech_dataset(config: SpeechTrainConfig) -> "Dataset":
    if config.use_demo_dataset:
        return build_demo_dataset(config)

    from datasets import Audio, load_dataset

    ds = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        trust_remote_code=True,
    )

    ds = ds.cast_column(
        config.dataset_audio_column,
        Audio(sampling_rate=config.mimi_sample_rate),
    )

    ds = ds.filter(
        _row_passes_filters,
        fn_kwargs={"config": config},
        desc="Filter Turkish speech rows",
    )

    if config.max_samples is not None and len(ds) > 0:
        n = min(config.max_samples, len(ds))
        ds = ds.select(range(n))
    return ds

def train_val_split(
    dataset: "Dataset", val_fraction: float, seed: int
) -> tuple["Dataset", "Dataset"]:
    if val_fraction <= 0 or len(dataset) < 2:
        return dataset, dataset.select(range(min(1, len(dataset))))
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]