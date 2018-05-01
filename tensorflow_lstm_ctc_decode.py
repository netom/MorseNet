#!/usr/bin/env python3

from __future__ import generator_stop

import time

# On my setup an annoying, but benign warning keeps appearing, messing
# with my visual cortex impeding testing.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import sys
import scipy.io.wavfile
import os

from config import *

from tensorflow_lstm_ctc_train import *

def main(args):

    if len(args) < 2:
        print("Usage: ./tensorflow_lstm_ctc_decode.py <filename.wav>")
        exit()

    rate, data = scipy.io.wavfile.read(args[1])
    timesteps = len(data) // CHUNK
    data = np.asarray(data[:timesteps * CHUNK], dtype=np.float32)
    data = (data - np.mean(data)) / np.std(data)
    
    # For prediction (or decoding), a separate,
    # smaller estimator is created
    estimator = tf.estimator.Estimator(
        model_fn=cw_model,
        model_dir='./model_use',
        params={
            'max_timesteps': timesteps,
            'batch_size': 1,
            'num_features': CHUNK,
            'input_layer_depth': 0,
            'input_layer_width': CHUNK,
            'recurrent_layer_depth': 2,
            'recurrent_layer_width': 128,
            'output_layer_depth': 1,
            'output_layer_width': 128
        }
    )

    def wav_input_fn():
        return tf.data.Dataset.from_tensors((tf.reshape(data, (timesteps,CHUNK)), []))

    res = estimator.predict(
        input_fn=wav_input_fn
    )
    for r in res:
        decoded_str = list(map(lambda x: MORSE_CHR[x], r['decoded']))
        print(''.join(decoded_str))

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.run(main)
