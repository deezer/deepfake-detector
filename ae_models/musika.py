"""
Musika
"""

import os
import sys

import numpy as np
import tensorflow as tf
import torch

cur_dir = os.getcwd()
sys.path.append(os.path.join(cur_dir, "pretrained/musika"))

from ae_models.ae import AE
from pretrained.musika.models import Models_functions
from pretrained.musika.parse.parse_decode import parse_args
from pretrained.musika.utils import Utils_functions

# import tensorflow as tf

checkpoint = os.path.join(cur_dir, "pretrained/musika/checkpoints/techno")
ae_path = os.path.join(cur_dir, "pretrained/musika/checkpoints/ae")


class Musika_ae(AE):
    def __init__(self):
        super().__init__("Musika")

        args = parse_args()
        self.args = args

        args.load_path = checkpoint
        args.dec_path = ae_path
        # args.mixed_precision = False

        M = Models_functions(args)
        self.U = Utils_functions(args)
        self.models_ls = M.get_networks()

    def encode(self, x):
        x = x.cpu()
        return self.encode_audio(x.T.numpy(), models_ls=self.models_ls)

    def decode(self, z):
        return torch.Tensor(
                self.U.decode_waveform(z[None, None, :,:], self.models_ls[3], self.models_ls[5], batch_size=64)
            ).T

    def encode_audio(self, audio, models_ls=None):
        critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

        time_compression_ratio = 16  # TODO: infer time compression ratio
        shape2 = self.args.shape
        wv = np.squeeze(audio)

        if wv.shape[0] > self.args.hop * self.args.shape * 2 + 3 * self.args.hop:

            rem = (wv.shape[0] - (3 * self.args.hop)) % (self.args.shape * self.args.hop)

            if rem != 0:
                wv = tf.concat([wv, tf.zeros([rem,2], dtype=tf.float32)], 0)

            chls = []
            for channel in range(2):

                x = wv[:, channel]
                x = tf.expand_dims(tf.transpose(self.U.wv2spec(x, hop_size=self.args.hop), (1, 0)), -1)
                ds = []
                num = x.shape[1] // self.args.shape
                rn = 0
                for i in range(num):
                    ds.append(
                        x[:, rn + (i * self.args.shape) : rn + (i * self.args.shape) + self.args.shape, :]
                    )
                del x
                ds = tf.convert_to_tensor(ds, dtype=tf.float32)
                lat = self.U.distribute_enc(ds, enc)
                del ds
                lat = tf.split(lat, lat.shape[0], 0)
                lat = tf.concat(lat, -2)
                lat = tf.squeeze(lat)



                ds2 = []
                num2 = lat.shape[-2] // shape2
                rn2 = 0
                for j in range(num2):
                    ds2.append(lat[rn2 + (j * shape2) : rn2 + (j * shape2) + shape2, :])
                ds2 = tf.convert_to_tensor(ds2, dtype=tf.float32)
                lat = self.U.distribute_enc(tf.expand_dims(ds2, -3), enc2)
                del ds2
                lat = tf.split(lat, lat.shape[0], 0)
                lat = tf.concat(lat, -2)
                lat = tf.squeeze(lat)
                chls.append(lat)

            lat = tf.concat(chls, -1)

            del chls
            return lat