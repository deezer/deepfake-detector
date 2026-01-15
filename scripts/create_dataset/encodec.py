import os
import sys
from glob import glob

import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from loader.global_variables import HOME
from ae_models.encodec import Encodec
from ae_models.pipeline import Pipeline

GPU = 0
encodec_conf = {
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

all_mp3_paths = glob(os.path.join(encodec_conf["DB_PATH"], "**/*.mp3"))
print("Found {} mp3 paths".format(len(all_mp3_paths)))

torch.cuda.set_device(GPU)

## load model

encodec_24 = Encodec(24, encodec_conf["SR"], encodec_conf["DEVICE"])
encodec_target_bandwidth = [3, 6, 24]

models = [ encodec_24 ]
out_name = [ 'encodec' ]

## main loop

pipeline = Pipeline(all_mp3_paths, encodec_conf)
pipeline.run_loop(models, out_name, encodec_target_bandwidth, has_cpu_preprocess = True)
