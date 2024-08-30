"""
Just a simple CNN that directly reads on the audio file / spectrogram.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv1D, Conv2D, Dense,
                                     Dropout, Flatten,
                                     GlobalMaxPooling1D,
                                     GlobalMaxPooling2D,
                                     GlobalAveragePooling2D,
                                     GlobalAveragePooling1D,
                                     Input, LeakyReLU,
                                     MaxPooling1D, MaxPooling2D)
from tensorflow.keras.models import Model

cnn_default_params = {
    "hidden": [ 128, 128, 256, 256, 512, 512 ],
    "strides": [  1,   3,   3,   3,   3,   3 ],
    "kernel":  [  3,   3,   3,   3,   3,   3 ] ,
    "conv_activation": "relu",
    "use_batch_norm": True,
    "head_dense": 64,
    "dense_activation": "relu",
    "global_pooling": "max",

    "sr": 44100,
    "audio_slice": 0.5,
}

class SimpleCNN():
    def __init__(self, input_shape, params = {}, detect_encoder = False):
        self.name = "simple"

        self.params = cnn_default_params
        self.params.update(params)

        X = Input((input_shape))
        Y = X

        for h, s, k in zip(self.params["hidden"], self.params["strides"], self.params["kernel"]):
            if self.params['use_maxpool']:
                Y = Conv1D(h, kernel_size=k,
                        padding='same', activation=self.params["conv_activation"])(Y)
                Y = MaxPooling1D(pool_size=s)(Y)
            else:
                Y = Conv1D(h, kernel_size=k, strides=s,
                        padding='same', activation=self.params["conv_activation"])(Y)

            if self.params["use_batch_norm"]:
                Y = BatchNormalization()(Y)

        if self.params["global_pooling"] == "average":
            Y = GlobalAveragePooling1D()(Y)
        else:
            Y = GlobalMaxPooling1D()(Y)

        Y = Dropout(0.2)(Y)

        if self.params["head_dense"] > 0:
            self.fc = Dense(self.params["head_dense"], activation = self.params["dense_activation"])
            Y = self.fc(Y)
            Y = Dropout(0.2)(Y)

        self.fc_out = Dense(1, activation = 'sigmoid', name="deepfake")

        if not detect_encoder:
            Y = self.fc_out(Y)
        else:
            self.fc_encoder = Dense(detect_encoder, activation = 'softmax', name="encoder")
            Y1 = self.fc_out(Y, )
            Y2 = self.fc_encoder(Y)
            Y = [Y1, Y2]

        self.m = Model(X, Y)



class SimpleSpectrogramCNN():
    def __init__(self, input_shape, params = {}, detect_encoder = False, get_embedder = False):
        self.name = "specnn"

        self.params = cnn_default_params
        self.params.update(params)

        X = Input((input_shape))
        Y = X

        for h, s, k in zip(self.params["hidden"], self.params["strides"], self.params["kernel"]):
            if self.params['use_maxpool']:
                Y = Conv2D(h, kernel_size=k, padding='same',
                            activation=self.params["conv_activation"])(Y)
                Y = MaxPooling2D(pool_size=s, padding='valid')(Y)
            else:
                Y = Conv2D(h, kernel_size=k, strides=s,
                            padding='same', activation=self.params["conv_activation"])(Y)
            if self.params["use_batch_norm"]:
                Y = BatchNormalization()(Y)

        if self.params["global_pooling"] == "average":
            Y = GlobalAveragePooling2D()(Y)
        else:
            Y = GlobalMaxPooling2D()(Y)

        if get_embedder:
            self.embedder = Model(X, Y, name = "embedder")

        Y = Dropout(0.2)(Y)

        if self.params["head_dense"] > 0:
            self.fc = Dense(self.params["head_dense"], activation = self.params["dense_activation"])
            Y = self.fc(Y)
            Y = Dropout(0.2)(Y)

        self.fc_out = Dense(1, activation = 'sigmoid', name="deepfake")

        if not detect_encoder:
            Y = self.fc_out(Y)
        else:
            self.fc_encoder = Dense(detect_encoder, activation = 'softmax', name="encoder")
            Y1 = self.fc_out(Y, )
            Y2 = self.fc_encoder(Y)
            Y = [Y1, Y2]

        self.m = Model(X, Y)
