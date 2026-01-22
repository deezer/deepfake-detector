import os
from glob import glob

import torch

from ae_models.dac import Lac_ae
from ae_models.pipeline import Pipeline

GPU = 2
lac_conf = {
    "DEVICE": "cuda",
    "DB_PATH": HOME+"/fma_medium",
    "OUT_DB": HOME+"/fma_rebuilt_medium",
    "BR_PATH": HOME+"/data/bitrates_ffmpeg_medium.npy",
    "DATA_SR": 44100,
    "SR": 44100, # target sr
    "MIN_DURATION": 3, # seconds
    "MAX_DURATION": 40,
    # "VERBOSE": True,
}

all_mp3_paths = glob(os.path.join(lac_conf["DB_PATH"], "**/*.mp3"))
print("Found {} mp3 paths".format(len(all_mp3_paths)))

torch.cuda.set_device(GPU)

## init

pipeline = Pipeline(all_mp3_paths, lac_conf)
lac_ae = Lac_ae(lac_conf["SR"], device = lac_conf["DEVICE"])

models = [lac_ae]
out_name = ['lac']

pipeline.run_loop(models, out_name, [2, 7, 14])