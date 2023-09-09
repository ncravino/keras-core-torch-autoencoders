#!/usr/bin/env python3

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras_core as keras

ENCODER_MODEL_FILE = "../model/encoder.keras"
DECODER_MODEL_FILE = "../model/decoder.keras"

encoder = keras.saving.load_model(ENCODER_MODEL_FILE)
decoder = keras.saving.load_model(DECODER_MODEL_FILE)

with np.load("./dataset/mnist.npz") as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype('float32') / 255.
x_validation = x_test[5001:10000].astype('float32') / 255.

print("reconstructing")

latent = encoder.predict(x_validation)
reconstructions = decoder.predict(latent)

print("printing")

plt.figure(figsize=(20, 4))
plt.gray()
n = 20
for i in range(n):
    val_plt = plt.subplot(2, n, i + 1)
    recon_plt = plt.subplot(2, n, i + 1 + n)

    val_plt.get_xaxis().set_visible(False)
    val_plt.get_yaxis().set_visible(False)
    recon_plt.get_xaxis().set_visible(False)
    recon_plt.get_yaxis().set_visible(False)
    
    val_plt.imshow(x_validation[i].reshape(28, 28))       
    recon_plt.imshow(reconstructions[i].reshape(28, 28))

plt.show()