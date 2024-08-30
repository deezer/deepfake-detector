"""
Training code

example:
python -m scripts.final.train --config specnn_amplitude --gpu 0
"""

import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras

from loader.global_variables import *
from loader.audio import AudioLoader, Augmenter
from loader.config import ConfLoader
from model.simple_cnn import SimpleCNN, SimpleSpectrogramCNN

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="gpu to evaluate on", type=int, default=-1)
parser.add_argument("--config", help="config file", type=str, default="specnn")
parser.add_argument("--codec", help="codec", type=str, default="")
parser.add_argument("--weights", help="continue training or fine-tune", type=str, default="")
parser.add_argument("--encoder", help="train on only one encoder", type=str, default="")
parser.set_defaults(external_home=False)
args = parser.parse_args()

GPU = args.gpu
ENCODER = ''
if args.encoder:
    ENCODER = args.encoder

configuration = ConfLoader(CONF_PATH)
configuration.load_model(args.config)
global_conf = configuration.conf

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU], 'GPU')
tf.config.experimental.set_memory_growth(gpus[GPU], True)


## utils

def save_conf(*confs, name = 'model'):
    global_conf = {}
    for conf in confs:
        global_conf.update(conf)
    np.save( os.path.join(WEIGHTS_PATH, '{}_logconf'.format(name)), global_conf)

@tf.function
def one_hot_encoder(y, depth):
    """ from idx to one hot """
    y1, y2 = y
    y1 = tf.cast(y1, tf.int32)
    idx = (1 - y1) * (y2 + 1)
    return y1, tf.one_hot(idx, depth)


## load data

loader = AudioLoader(POS_DB_PATH, NEG_DB_PATH, global_conf, split_path = SPLIT_PATH)
augmenter = Augmenter(global_conf)

if "patch_size_t" in global_conf:
    it_train = loader.create_patch_iterator(augmenter, 'train')
    it_val = loader.create_patch_iterator(augmenter, 'validation')
else:
    it_train = loader.create_tf_iterator('train', augmenter = augmenter, encoder = ENCODER)
    it_val = loader.create_tf_iterator('validation', augmenter = augmenter, encoder = ENCODER)

if not ENCODER:  # learn to detect the encoders
    it_train = it_train.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders+1)) )
    it_val = it_val.map(lambda x, y: (x, one_hot_encoder(y, loader.n_encoders+1)))
else:   # just real/fake
    it_train = it_train.map(lambda x, y: (x, y[0]) )
    it_val = it_val.map(lambda x, y: (x, y[0]))

if 'use_raw' in global_conf:
    it_train = it_train.map(lambda x, y: (tf.expand_dims(x, -1), y) )
    it_val = it_val.map(lambda x, y: (tf.expand_dims(x, -1), y) )

_it = iter(it_train.take(11))
input_batch, y_batch = next(_it)


## train

n_encoders = loader.n_encoders+1
input_size = input_batch.shape[1:]
if "patch_size_f_min" in global_conf:
    input_size = (None, input_size[1], input_size[2])

if ENCODER:
    print("\nEncoder-specific model! `{}`\n".format(ENCODER))
    n_encoders = None
if 'use_raw' in global_conf:
    model = SimpleCNN(input_size, global_conf, detect_encoder = n_encoders)
else:
    model = SimpleSpectrogramCNN(input_size, global_conf, detect_encoder = n_encoders)
model.m.summary()

if args.weights != '':
    print("\nStarting training from checkpoint", args.weights)
    model.m.load_weights(args.weights)
    model.m.compile( optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3) )

if not ENCODER:
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss = {
            'deepfake': keras.losses.BinaryCrossentropy(),
            'encoder': keras.losses.CategoricalCrossentropy(),
        },
        loss_weights = { 'deepfake': 1.0, 'encoder': 0.2 },
        metrics = {
            'deepfake': keras.metrics.BinaryAccuracy(),
            'encoder': keras.metrics.CategoricalAccuracy(),
        },
    )
else:
    model.m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss = { 'deepfake': keras.losses.BinaryCrossentropy() },
        metrics = { 'deepfake': keras.metrics.BinaryAccuracy() },
    )

if not ENCODER:
    save_conf(global_conf, loader.params, model.params, augmenter.params, name = args.config)

saver = tf.keras.callbacks.ModelCheckpoint( "{}/{}{}".format(WEIGHTS_PATH, global_conf['name'], ENCODER), monitor="val_loss", verbose=1,)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)
reducer = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss',
                patience = 10,
                factor=0.1, mode='auto', min_lr=1e-5)

model.m.fit(it_train, epochs = 200, steps_per_epoch = 100 * global_conf["repeat"],
                verbose = 1,
                validation_data = it_val, validation_steps = 10 * global_conf["repeat"],
                callbacks = [ early_stop, reducer, saver ] )
