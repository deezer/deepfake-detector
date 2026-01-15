import os
from glob import glob

import torch

from ae_models.identity import Identity
from ae_models.pipeline import Pipeline

identity_conf = {
    "DEVICE": "cpu",
    "DB_PATH": HOME+"/fma_medium",
    "OUT_DB": HOME+"/fma_rebuilt_medium",
    "BR_PATH": HOME+"/deepfake/data/bitrates_ffmpeg_medium.npy",
    "DATA_SR": 44100,
    "SR": 44100, # target sr
    "MIN_DURATION": 3, # seconds
    "MAX_DURATION": 40,
    # "VERBOSE": True,
}

all_mp3_paths = glob(os.path.join(identity_conf["DB_PATH"], "**/*.mp3"))
print("Found {} mp3 paths".format(len(all_mp3_paths)))


## load model

models = [ Identity() ]
out_name = [ 'resampled' ]

##  loop

pipeline = Pipeline(all_mp3_paths, identity_conf)
pipeline.run_loop(models, out_name)
