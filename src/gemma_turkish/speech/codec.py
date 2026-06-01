"""Kyutai Mimi codec wrapper (encode teacher speech → discrete tokens)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, MimiModel

@dataclass
class EncodedSpeech:
    """Mimi RVQ codes: (batch, num_codebooks, num_frames)."""
    audio_codes: torch.LongTensor
    num_frames: int


class MimiSpeechCodec(nn.Module):
    """Frozen Mimi encoder/decoder matching Frisson's 8-codebook setup."""

    def __init__(
        self,
        model_id: str = "kyutai/mimi",
        num_codebooks: int = 8,
        sample_rate: int = 24000,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.num_codebooks = num_codebooks
        self.sample_rate = sample_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.codec = MimiModel.from_pretrained(model_id)
        self.codebook_size = int(self.codec.config.codebook_size)
        if device is not None:
            self.to(device)
        self.eval()

    @property
    def frame_rate_hz(self) -> float:
        # Blog: 12.5 Hz; HF config may expose frame rate via sampling / hop.
        sr = float(self.codec.config.sampling_rate)
        # Mimi downsampling product from config (8*6*5*4 = 960) → sr/960
        ratios = getattr(self.codec.config, "upsampling_ratios", [8, 6, 5, 4])
        hop = 1
        for r in ratios:
            hop *= int(r)
        return sr / hop

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int | None = None,
    ) -> EncodedSpeech:
        """Encode mono waveform (samples,) or (1, samples) to discrete codes."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        sr = sample_rate or self.sample_rate
        inputs = self.feature_extractor(
            raw_audio=waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
        )
        # Frozen Mimi stays on CPU under HF Trainer; CUDA inputs + CPU weights break cdist.
        if next(self.codec.parameters()).device.type != "cpu":
            self.codec.cpu()
        input_values = inputs["input_values"]
        padding_mask = inputs.get("padding_mask")

        encode_out = self.codec.encode(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=self.num_codebooks,
        )
        codes = (
            encode_out.audio_codes
            if hasattr(encode_out, "audio_codes")
            else encode_out[0]
        )
        codes = codes.long()
        return EncodedSpeech(audio_codes=codes, num_frames=codes.shape[-1])

    @torch.inference_mode()
    def decode_codes(self, audio_codes: torch.LongTensor) -> torch.Tensor:
        """Codes (B, K, T) → waveform (B, 1, samples)."""
        codec_device = next(self.codec.parameters()).device
        if audio_codes.device != codec_device:
            audio_codes = audio_codes.to(codec_device)
        out = self.codec.decode(audio_codes)
        if hasattr(out, "audio_values"):
            return out.audio_values
        return out[0]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
