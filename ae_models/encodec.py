"""
Encodec
"""

import torch
from transformers import AutoProcessor, EncodecModel

from ae_models.ae import AE


class Encodec(AE):
    """
    Griffin-lim + mel scale inverter
    """

    def __init__(self, bandwidth, sr, device="cuda"):
        super().__init__("encodec")

        self.bandwidth = bandwidth
        self.sr = sr
        self.device = device

        self.model = EncodecModel.from_pretrained("facebook/encodec_48khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

        self.model.eval()
        self.model.to(self.device)


    def preprocesss(self, x):
        return self.processor(raw_audio=x, sampling_rate=self.processor.sampling_rate, return_tensors="pt")


    def encode(self, x, padding_mask):
        return self.model.encode(x, padding_mask, bandwidth = self.bandwidth)


    def decode(self, z, scales, padding_mask):
        audio_values = self.model.decode(z,
                scales,
                padding_mask,
                )[0]
        return audio_values


    def autoencode(self, x):
        inputs = self.processor(raw_audio=x, sampling_rate=self.processor.sampling_rate, return_tensors="pt")

        inputs = inputs.to(self.device)
        # explicitly encode then decode the audio inputs
        encoder_outputs = self.model.encode(
                inputs["input_values"], inputs["padding_mask"],
                bandwidth = self.bandwidth,  # 3, 6, 12, 24
                )
        audio_values = self.model.decode(encoder_outputs.audio_codes,
                encoder_outputs.audio_scales,
                inputs["padding_mask"],
                )[0]
        return audio_values[:,:,:x.shape[1]]


    def autoencode_multi(self, x, codec):
        inputs = self.processor(raw_audio=x, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        inputs = inputs.to(self.device)
        # explicitly encode then decode the audio inputs
        encoder_outputs = self.model.encode(
                inputs["input_values"], inputs["padding_mask"],
                bandwidth = self.bandwidth,  # 3, 6, 12, 24
                )
        audio_vals = encoder_outputs["audio_codes"]

        decoded_audio = []
        for c in codec:
            num_codebooks = (c // 3) * 2
            audio_vals_target = audio_vals[:,:,:num_codebooks]
            audio_rebuilt = self.model.decode(audio_vals_target,
                    encoder_outputs["audio_scales"],
                    inputs["padding_mask"],
                    )[0]
            decoded_audio.append(torch.squeeze(audio_rebuilt))

        return decoded_audio
