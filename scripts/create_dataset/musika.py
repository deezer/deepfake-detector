import os
from glob import glob

import tensorflow as tf
import torch

from ae_models.musika import Musika_ae
from ae_models.pipeline import Pipeline

GPU = 0

torch.cuda.set_device(GPU)
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU], 'GPU')
tf.config.experimental.set_memory_growth(gpus[GPU], True)

musika_conf = {
    "DEVICE": "cuda",
    "DB_PATH": HOME+"/fma_medium",
    "OUT_DB": HOME+"/fma_rebuilt_medium",
    "BR_PATH": HOME+"/deepfake/data/bitrates_ffmpeg_medium.npy",
    "DATA_SR": 44100,
    "SR": 44100, # target sr
    "MIN_DURATION": 3, # seconds
    "MAX_DURATION": 40,
    # "VERBOSE": True,
}

all_mp3_paths = glob(os.path.join(musika_conf["DB_PATH"], "**/*.mp3"))
print("Found {} mp3 paths".format(len(all_mp3_paths)))

torch.cuda.set_device(GPU)

## load model

musika = Musika_ae()

models = [ musika ]
out_name = ['musika']

##

pipeline = Pipeline(all_mp3_paths, musika_conf)
pipeline.run_loop(models, out_name)