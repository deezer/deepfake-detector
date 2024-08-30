"""
Musika
"""

import os
import sys

import torch

cur_dir = os.getcwd()
sys.path.append(os.path.join(cur_dir, "pretrained/musika"))

from ae_models.ae import AE
from pretrained.musika.models import Models_functions
from pretrained.musika.parse.parse_decode import parse_args
from pretrained.musika.utils import Utils_functions
from pretrained.musika.utils_encode import UtilsEncode_functions

# import tensorflow as tf

checkpoint = os.path.join(cur_dir, "pretrained/musika/checkpoints/techno")
ae_path = os.path.join(cur_dir, "pretrained/musika/checkpoints/ae")


class Musika_ae(AE):
    def __init__(self):
        super().__init__("Musika")

        args = parse_args()

        args.load_path = checkpoint
        args.dec_path = ae_path
        # args.mixed_precision = False

        M = Models_functions(args)
        self.U = Utils_functions(args)
        self.UE = UtilsEncode_functions(args)
        self.models_ls = M.get_networks()

    def encode(self, x):
        x = x.cpu()
        return self.UE.encode_audio(x.T.numpy(), models_ls=self.models_ls)

    def decode(self, z):
        return torch.Tensor(
                self.U.decode_waveform(z[None, None, :,:], self.models_ls[3], self.models_ls[5], batch_size=64)
            ).T