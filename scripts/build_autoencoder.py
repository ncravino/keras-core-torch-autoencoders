#!/usr/bin/env python3

import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import keras_core as keras
from keras_core import layers
from keras_core import ops
from keras_core import regularizers

ENCODER_MODEL_FILE = "../model/encoder.keras"
DECODER_MODEL_FILE = "../model/decoder.keras"

input_dim = 28

with np.load("./dataset/mnist.npz") as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype('float32') / 255.
x_test = x_test[0:5000].astype('float32') / 255.

print (x_train.shape)
input_layer = layers.Input(shape=(input_dim,input_dim,1), name="input")

encoder = keras.Sequential(
    [         
        input_layer,                    
        layers.Conv2D(8, (4, 4), activation='leaky_relu', padding='same'),              
        layers.MaxPooling2D((2,2)),                        
        layers.Dense(8, use_bias=True, activation='leaky_relu', activity_regularizer=regularizers.L1(10e-3), bias_regularizer=regularizers.L1(10e-3)),
        layers.Dense(16, use_bias=True, activation='leaky_relu', activity_regularizer=regularizers.L1(10e-3), bias_regularizer=regularizers.L1(10e-3)),
        layers.Dense(2, use_bias=True, activation='leaky_relu', activity_regularizer=regularizers.L1(10e-3), bias_regularizer=regularizers.L1(10e-3)),
    ], name="encoder"
)

decoder = keras.Sequential(    [
        layers.UpSampling2D((2,2)),
        layers.Dense(20, use_bias=True, activation='leaky_relu', activity_regularizer=regularizers.L1(10e-3), bias_regularizer=regularizers.L1(10e-3)),
        layers.Dense(10, use_bias=True, activation='leaky_relu', activity_regularizer=regularizers.L1(10e-3), bias_regularizer=regularizers.L1(10e-3)),
        layers.Conv2D(1, (4, 4), activation='sigmoid', padding='same')
    ], name="decoder"
)

model = keras.Sequential(
    [
        input_layer, 
        encoder,
        decoder
    ]
)

model.summary()

model.compile(optimizer="adam", loss=keras.losses.Huber(
    delta=1.0, reduction="sum_over_batch_size", name="huber_loss")
)

model.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

keras.saving.save_model(encoder, ENCODER_MODEL_FILE)
keras.saving.save_model(decoder, DECODER_MODEL_FILE)

#235/235 ━━━━━━━━━━━━━━━━━━━━ 8s 32ms/step - loss: 2.5516e-04 - val_loss: 2.3554e-04