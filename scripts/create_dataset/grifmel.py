import os
import sys
from glob import glob

import torch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from loader.global_variables import HOME
from ae_models.griffinmel import GriffinMel
from ae_models.pipeline import Pipeline

GPU = 0
griffin_conf = {
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

all_mp3_paths = glob(os.path.join(griffin_conf["DB_PATH"], "**/*.mp3"))
print("Found {} mp3 paths".format(len(all_mp3_paths)))

torch.cuda.set_device(GPU)

## load model

riffusion_params = {
    "n_fft": int(400 / 1000 * griffin_conf["SR"]), # /SR = 46ms
    "hop_fft": int( 10 / 1000 * griffin_conf["SR"]),
    "win_fft": int(100 / 1000 * griffin_conf["SR"]),

    "griffin_iter": 32,

    "n_mels": 512,
}

riffusion_256 = dict(riffusion_params)
riffusion_256["n_mels"] = 256

ae_griffinmel_512 = GriffinMel(riffusion_params, griffin_conf["SR"], griffin_conf["DEVICE"])
ae_griffinmel_256 = GriffinMel(riffusion_256, griffin_conf["SR"], griffin_conf["DEVICE"])

models = [ae_griffinmel_512, ae_griffinmel_256]
out_name = ['griffin512', 'griffin256']

##  loop

pipeline = Pipeline(all_mp3_paths, griffin_conf)
pipeline.run_loop(models, out_name)
