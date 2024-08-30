"""
GriffinLim
"""

import numpy as np
import torch
import torchaudio

from ae_models.ae import AE

default_params = {
    "n_fft": 1024,
    "hop_fft": 256,
    "win_fft": 512,

    "griffin_iter": 32,

    "n_mels": 128
}

class GriffinMel(AE):
    """
    Griffin-lim + mel scale inverter
    """

    def __init__(self, params, sr, device="cuda"):
        super().__init__("mel-griffin")

        self.params = dict()
        self.params.update(default_params)
        self.params.update(params)

        self.sr = sr
        self.device = device

        self.audio2spec = torchaudio.transforms.Spectrogram(
                n_fft = self.params["n_fft"],
                hop_length = self.params["hop_fft"],  # pas temporel de hop/sr
                win_length = self.params["win_fft"],
                window_fn=torch.hann_window,
                power = None,
            )
        self.audio2spec.to(device=torch.device(self.device), dtype=torch.float32)

        self.spec2audio = torchaudio.transforms.GriffinLim(
                n_fft=self.params["n_fft"],
                n_iter=self.params["griffin_iter"],
                win_length=self.params["win_fft"],
                hop_length=self.params["hop_fft"],
                window_fn=torch.hann_window,
                power = 1,
            )
        self.spec2audio.to(device=torch.device(self.device), dtype=torch.float32)

        self.mel_scaler = torchaudio.transforms.MelScale(
                n_mels=self.params["n_mels"],
                sample_rate=self.sr,
                n_stft=self.params["n_fft"] // 2 + 1,
                mel_scale = 'htk',  # slaney
            )
        self.mel_scaler.to(device=torch.device(self.device), dtype=torch.float32)

        self.mel_matrix = self.mel_scaler.fb
        self.inv_mel_matrix = torch.linalg.pinv(self.mel_matrix).T

    def inverse_mel_scaler(self, mels_spec):
        return self.inv_mel_matrix @ mels_spec

    def _encode_mono(self, x):
        return self.mel_scaler(torch.abs(self.audio2spec(x)))

    def encode(self, x):
        return self.map_stack(x, self._encode_mono)

    def _decode_mono(self, z):
        return self.spec2audio(self.inverse_mel_scaler(z))

    def decode(self, z):
        return self.map_stack(z, self._decode_mono)
