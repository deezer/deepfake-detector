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
parser.add_argument("--codec", help="codec", type=str, default="")
parser.add_argument('--external_home', action='store_true')
args = parser.parse_args()

if not args.weights:
    args.weights = args.config
GPU = args.gpu
CODEC = ''
CODEC_EXTENSION = ''
if args.codec != "":
    CODEC_EXTENSION = args.codec
    CODEC = CODEC_EXTENSION + "_64"
    print("\nEvaluating on codec {}!\n".format(CODEC))
    if args.codec == "opus":
        CODEC = 'libopus'
    POS_DB_PATH = CODEC_DB_PATH+CODEC+"/real"
    NEG_DB_PATH = CODEC_DB_PATH+CODEC+"/"
ENCODER = ''
if args.encoder:
    ENCODER = args.encoder

configuration = ConfLoader(CONF_PATH)
configuration.load_model(args.config)
global_conf = configuration.conf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU], 'GPU')
tf.config.experimental.set_memory_growth(gpus[GPU], True)

if args.repeat > 0:
    global_conf['repeat'] = args.repeat  # keep the repeat low

##

adversarial = None
# adversarial_params = {
#     "adversarial_effects": ["pitch"],
#     "sr": 44100,
#     "target_audio_slice": 2.0,
#     "adversarial_pitch": 3,
#     "adversarial_stretch": 0.2, # -> [80%, 120%]
# }
#
# if "stretch" in adversarial_params["adversarial_effects"]:
#     print("loading more audio for time streching!")
#     global_conf["audio_slice"] = adversarial_params["target_audio_slice"] * (1 + adversarial_params["adversarial_stretch"])
#
# adversarial = AdversarialAugmenter(adversarial_params)


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
if not ENCODER:
    it_test = it_test.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders+1)) )
else:
    it_test = it_test.map(lambda x, y: (x, y[0]) )

if 'use_raw' in global_conf:
    it_test = it_test.map(lambda x, y: (tf.expand_dims(x, -1), y) )

_it = iter(it_test)
input_batch, y_batch = next(_it)

## normal run

n_encoders = loader.n_encoders+1
if ENCODER:
    print("\nEncoder-specific model! `{}`\n".format(ENCODER))
    n_encoders = None
if 'use_raw' in global_conf:
    model = SimpleCNN(input_batch.shape[1:], global_conf, detect_encoder = n_encoders)
else:
    model = SimpleSpectrogramCNN(input_batch.shape[1:], global_conf, detect_encoder = n_encoders)

# model.m.summary()
model.m.load_weights( os.path.join(WEIGHTS_PATH, args.weights + args.encoder) )
if not ENCODER:
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss_weights = { 'deepfake': 1.0, 'encoder': 0.2 },
        metrics = {
            'deepfake': keras.metrics.BinaryAccuracy(),
            'encoder': keras.metrics.CategoricalAccuracy(),
        })
else:
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics = { 'deepfake': keras.metrics.BinaryAccuracy() },
    )

scores = {}

if not adversarial:
    print("\nALL:")
    scores['all'] = model.m.evaluate(it_test, batch_size = global_conf["batch_size"], steps=args.steps)

max_steps = min( len(loader.split_mp3['test']) // global_conf["batch_size"] * global_conf["repeat"] + 1,
            args.steps )


for encoder in loader.encoders:
    print("\nENCODER {}:".format(encoder))
    fake_it = loader.create_fast_eval_iterator(encoder, augmenter = augmenter, adversarial = adversarial)
    if not ENCODER:
        fake_it = fake_it.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders+1)) )
    else:
        fake_it = fake_it.map(lambda x, y: (x, y[0]))
    if 'use_raw' in global_conf:
        fake_it = fake_it.map(lambda x, y: (tf.expand_dims(x, -1), y) )
    scores[encoder] = model.m.evaluate(fake_it, batch_size = global_conf["batch_size"], steps=max_steps)

    np.save(os.path.join(RESULT_PATH, "{}{}{}.npy".format(args.config, CODEC_EXTENSION, ENCODER)), scores)

print("\nREAL:")
real_it = loader.create_fast_eval_iterator('real', augmenter = augmenter, adversarial = adversarial)
real_it = real_it.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders+1)) )
if 'use_raw' in global_conf:
    real_it = real_it.map(lambda x, y: (tf.expand_dims(x, -1), y) )
scores['real'] = model.m.evaluate(real_it, batch_size = global_conf["batch_size"], steps=max_steps)
np.save(os.path.join(RESULT_PATH, "{}{}{}.npy".format(args.config, CODEC_EXTENSION, ENCODER)), scores)