"""
DAC / LAC
"""

import os
import sys

import torch

cur_dir = os.getcwd()
sys.path.append(os.path.join(cur_dir, "pretrained/lac"))

from ae_models.ae import AE
from pretrained.lac.lac.model.lac import LAC

lac_ckpt_path = os.path.join(cur_dir, "pretrained/vampnet/codec.pth")

class Lac_ae(AE):
    def __init__(self, sr, device = "cuda"):
        super().__init__("LAC")

        self.sr = sr
        self.device = device
        self.model = LAC.load(lac_ckpt_path)
        self.model.eval()
        self.model.to(self.device)


    def encode(self, x):
        preprocess, _ = self.model.preprocess(x, self.sr)
        z = self.model.encode(preprocess[:, None, :], self.sr)
        return z['z']


    def decode(self, z):
        y = self.model.decode(z)['audio']
        y = torch.squeeze(y)
        return y

    def autoencode_multi(self, x, codec):
        preprocess, _ = self.model.preprocess(x, self.sr)
        z = self.model.encode(preprocess[:, None, :], self.sr)
        codes = z['codes']

        decoded_audio = []
        for c in codec:
            z_red = self.model.quantizer.from_codes(codes[:,:c,:])[0]
            r_audioraw = self.model.decode(z_red)['audio']
            decoded_audio.append(torch.squeeze(r_audioraw))

        return decoded_audio







