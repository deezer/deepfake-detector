"""
Simple evaluation code

example:
python -m scripts.final.eval --config specnn_amplitude --gpu 0 --steps 20 --repeat 5
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from loader.global_variables import *
from loader.audio import AudioLoader, EvalAugmenter, AdversarialAugmenter
from loader.config import ConfLoader
from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="config file", type=str, default="specnn")
parser.add_argument("--weights", help="weights file, else defaults to config", type=str)
parser.add_argument("--encoder", help="eval model trained on only one encoder", type=str, default="")
parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=-1)
parser.add_argument("--steps", help="gpu to evaluate on", type=int, default=500)
parser.add_argument("--repeat", help="gpu to evaluate on", type=int, default=-1)
parser.add_argument('--external_home', action='store_true')
args = parser.parse_args()

if not args.weights:
    args.weights = args.config
GPU = args.gpu
CODEC = ''
CODEC_EXTENSION = ''
N_BINS = 10

configuration = ConfLoader(CONF_PATH)
configuration.load_model(args.config)
global_conf = configuration.conf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU], 'GPU')
tf.config.experimental.set_memory_growth(gpus[GPU], True)

if args.repeat > 0:
    global_conf['repeat'] = args.repeat  # keep the repeat low


##

loader = AudioLoader(POS_DB_PATH, NEG_DB_PATH, global_conf, split_path = SPLIT_PATH,
        codec = CODEC_EXTENSION)
augmenter = EvalAugmenter(global_conf)

@tf.function
def one_hot_encoder(y, depth):
    y1, y2 = y
    y1 = tf.cast(y1, tf.int32)
    idx = (1 - y1) * (y2 + 1)
    return y1, tf.one_hot(idx, depth)

it_test = loader.create_tf_iterator('test', augmenter = augmenter)
it_test = it_test.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders+1)) )

if 'use_raw' in global_conf:
    it_test = it_test.map(lambda x, y: (tf.expand_dims(x, -1), y) )

_it = iter(it_test)
input_batch, y_batch = next(_it)


##

n_encoders = loader.n_encoders+1
if 'use_raw' in global_conf:
    model = SimpleCNN(input_batch.shape[1:], global_conf, detect_encoder = n_encoders)
else:
    model = SimpleSpectrogramCNN(input_batch.shape[1:], global_conf, detect_encoder = n_encoders)


model.m.load_weights( os.path.join(WEIGHTS_PATH, args.weights + args.encoder) )
model.m.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss_weights = { 'deepfake': 1.0, 'encoder': 0.2 },
    metrics = {
        'deepfake': keras.metrics.BinaryAccuracy(),
        'encoder': keras.metrics.CategoricalAccuracy(),
    })

## basic probability calibration

max_steps = min(
        len(loader.split_mp3['test']) // global_conf["batch_size"] * global_conf["repeat"] + 1,  args.steps )

all_res = []

for i in tqdm(range(max_steps)):
     input_batch, y = next(_it)
     p, q = model.m.predict(input_batch, batch_size = global_conf['batch_size'])
     all_res.append(np.concatenate((p, np.array(y[0])[:,None]), -1))

all_res = np.concatenate( all_res, 0)

np.save(os.path.join(RESULT_PATH, "calibration_{}.npy".format(args.config)), all_res)


## audio mixing calibration

it_test = loader.create_calibration_iterator('test', augmenter = augmenter, n_bins = N_BINS)
_it = iter(it_test)

max_steps = min(
        len(loader.split_mp3['test']) // global_conf["batch_size"] * N_BINS + 1,  args.steps )
all_res = []

for i in tqdm(range(max_steps)):
    input_batch, y = next(_it)
    p, q = model.m.predict(input_batch, batch_size = global_conf['batch_size'])
    all_res.append(np.concatenate((p, np.array(y)[:,None]), -1))

all_res = np.concatenate( all_res, 0)

np.save(os.path.join(RESULT_PATH, "mixing_{}.npy".format(args.config)), all_res)