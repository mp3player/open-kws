"""Shared audio settings for training and checkpoint inference."""
from dataclasses import asdict, dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    power: float = 2.0
    center: bool = False
    log_floor: float = 1e-10

    def __post_init__(self):
        if (not 0 < self.win_length <= self.n_fft or self.sample_rate <= 0
                or self.hop_length <= 0 or self.n_mels <= 0 or self.power <= 0
                or self.log_floor <= 0 or self.center):
            raise ValueError("Invalid audio config; this pipeline requires center=False.")

    def to_dict(self):
        return asdict(self)


class LogMel:
    def __init__(self, config=None):
        self.config = config or AudioConfig()
        values = self.config.to_dict()
        values.pop("log_floor")
        self.mel = torchaudio.transforms.MelSpectrogram(**values)

    def waveform(self, samples, sample_rate, channels_first=True):
        audio = torch.as_tensor(samples, dtype=torch.float32)
        if audio.ndim == 2:
            audio = audio.mean(dim=0 if channels_first else 1)
        if audio.ndim != 1 or audio.numel() == 0 or not torch.isfinite(audio).all():
            raise ValueError("Expected nonempty finite mono or multichannel audio.")
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if sample_rate != self.config.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.config.sample_rate)
        if len(audio) < self.config.n_fft:
            raise ValueError(f"Audio shorter than FFT window ({self.config.n_fft} samples).")
        return audio.contiguous()

    def __call__(self, samples, sample_rate, channels_first=True):
        audio = self.waveform(samples, sample_rate, channels_first)
        return self.mel(audio).clamp_min(self.config.log_floor).log().T.contiguous()
