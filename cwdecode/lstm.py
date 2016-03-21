#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

import os
import sys
import wave
import theano
import cPickle
import numpy as np
from config import *
import theano.tensor as T
import extensions

import blocks as bl
import blocks.bricks as br
import blocks.graph as blgraph
import blocks.bricks.cost as brcost
import blocks.initialization as blinit
import blocks.bricks.recurrent as brrec
import blocks.extensions.monitoring as lbmon
import blocks.extensions as blext
import blocks.algorithms as blalg
import blocks.extensions.monitoring as blmon
import blocks.main_loop as blml

from fuel.streams import DataStream
from fuel.datasets import IterableDataset

from collections import OrderedDict

import scipy.io.wavfile

N_CLASSES  = len(MORSE_CHR)

#
# "Dimension" of the audio data in the stream:
# - The number of chunks in the sequence
# - Batch size: The number of sequences in a batch
# - Chunk size: The number of samples in a chunk
#
def get_datastream(offset, num_batches):
    x = []
    y = []
    print "Loading %d batches with %d samples each..." % (num_batches, BATCH_SIZE)
    for i in xrange(num_batches):
        dirname = TRAINING_SET_DIR + '/%04d' % (offset + i)
        seq_length = int(open(dirname + '/config.txt').read().strip())
        x_b = np.zeros(((seq_length // CHUNK) + 1, BATCH_SIZE, CHUNK), dtype=np.float32)
        y_b = np.zeros((seq_length // CHUNK + 1, BATCH_SIZE), dtype=np.int64)
        for j in xrange(BATCH_SIZE):
            _, audio = scipy.io.wavfile.read(dirname + '/%03d.wav' % j)
            audio =  (audio / 2**13).astype(np.float32)

            padded_audio = np.pad(audio, (0, CHUNK - (len(audio) % CHUNK)), 'constant', constant_values=(0, 0))
            reshaped_padded_audio = padded_audio.reshape((len(padded_audio) // CHUNK, CHUNK))
            x_b[:,j,:] = reshaped_padded_audio

            f = open(dirname + '/%03d.txt' % j, 'r')
            lines = map(lambda line: line.split(','), filter(lambda line: line != '', map(lambda line: line.rstrip(), f.readlines())))
            chars = map(lambda rec: (MORSE_ORD[rec[0]], float(rec[1])), lines)
            f.close()

            for char in chars:
                y_b[char[1] * FRAMERATE // CHUNK][j] = char[0]

        sys.stdout.write("\rLoaded %d... " % (i+1))
        sys.stdout.flush()

        x.append(x_b)
        y.append(y_b)

    print "\nDone.\n"

    return DataStream(dataset=IterableDataset(OrderedDict([('x', x), ('y', y)])))

stream_train = get_datastream(0,   500)
stream_test  = get_datastream(500, 100)

x = T.ftensor3('x')
y = T.lmatrix('y')

input_layer = br.MLP(
    activations=[br.Rectifier(), br.Rectifier()],
    dims=[CHUNK, 256, 256],
    name='input_layer',
    weights_init=blinit.IsotropicGaussian(0.01),
    biases_init=blinit.Constant(0)
)
input_layer_app = input_layer.apply(x)
input_layer.initialize()

recurrent_layer = brrec.LSTM(
    dim=64,
    activation=br.Tanh(),
    name='recurrent_layer',
    weights_init=blinit.IsotropicGaussian(0.01),
    biases_init=blinit.Constant(0)
)
recurrent_layer_h, recurrent_layer_c = recurrent_layer.apply(input_layer_app)
recurrent_layer.initialize()

output_layer = br.MLP(
    activations=[br.Rectifier(), None],
    dims=[64, 64, N_CLASSES],
    name='output_layer',
    weights_init=blinit.IsotropicGaussian(0.01),
    biases_init=blinit.Constant(0)
)
output_layer_app = output_layer.apply(recurrent_layer_h)
output_layer.initialize()

y_hat_flat = br.Softmax().apply(output_layer_app.reshape((output_layer_app.shape[0]*output_layer_app.shape[1], output_layer_app.shape[2])))

y_flat = T.flatten(y)

cost = brcost.CategoricalCrossEntropy().apply(y_flat, y_hat_flat)
cost.name = 'cost'

y_hat_flat_onehot = T.argmax(y_hat_flat, axis=1)

characters                                    = T.switch(T.neq(y_flat, 0), np.float32(1.0), np.float32(0))
chunks_gotten_right                           = T.switch(T.eq(y_hat_flat_onehot, y_flat), np.float32(1.0), np.float32(0.0))
classification_success_percent                = T.cast(T.sum(chunks_gotten_right), 'float32') / y_flat.shape[0] * np.float32(100.0)
classification_success_percent.name           = 'classification_success_percent'
character_classification_success_percent      = T.sum(chunks_gotten_right * characters) / T.sum(characters) * np.float32(100.0)
character_classification_success_percent.name = 'character_classification_success_percent'

cg = blgraph.ComputationGraph(cost)

# Load saved model if exists (BUG)

savefname = "saved_params/lstm.pickle"
if os.path.exists(savefname):
    print "\n*** *** *** R E S U M I N G   T R A I N I N G *** *** ***\n\n"
    with open(savefname, "r") as f:
        values = cPickle.load(f)

    parameters = extensions.get_parameters([input_layer, recurrent_layer, output_layer])

    for parameter_name in values:
        parameters[parameter_name].set_value(values[parameter_name])

algorithm = blalg.GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=blalg.Adam(learning_rate=0.0005)
)

test_monitor = blmon.DataStreamMonitoring(
    variables=[
        cost,
        classification_success_percent,
        character_classification_success_percent
    ],
    data_stream=stream_test,
    prefix='test'
)

train_monitor = blmon.TrainingDataMonitoring(
    variables=[
        cost,
        classification_success_percent,
        character_classification_success_percent
    ],
    prefix='train',
    after_epoch=True
)

main_loop = blml.MainLoop(algorithm, stream_train,
    extensions=[
        test_monitor,
        train_monitor,
        blext.FinishAfter(after_n_epochs=10000),
        blext.Printing(),
        blext.ProgressBar(),
        extensions.SaveBestModel("saved_params/lstm.pickle", [input_layer, recurrent_layer, output_layer])
    ]
)  

main_loop.run()

