#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

import sys
import theano
import cPickle
import numpy as np
from config import *
import theano.tensor as T

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

N_CLASSES  = len(MORSE_CHR)

#
# "Dimension" of the data stream:
# Number of batches: This stream contains this many batches
# Sequence length: The length of a sequence (SAMPLE_CUNKS)
# Batch size: The numnber of sequences in a batch
# Chunk size: The number of samples in a chunk
#
def get_datastream(num_batches, offset):
    x = []
    y = []
    print "Loading %d batches..." % num_batches
    for i in xrange(num_batches):
        with open('training_set/batch_%d.pickle' % (offset + i), 'r') as f:
            x_b, y_b = cPickle.load(f)
        sys.stdout.write("\rLoaded %d... " % i)
        sys.stdout.flush()
        x.append(x_b)
        y.append(y_b)
    print "\nDone.\n"

    return DataStream(dataset=IterableDataset(OrderedDict([('x', x), ('y', y)])))

stream_train = get_datastream(80, 0)
stream_test  = get_datastream(20, 80)

x = T.ftensor3('x')
y = T.lmatrix('y')

input_layer = br.MLP(
    activations=[br.Rectifier(), br.Rectifier()],
    dims=[CHUNK, CHUNK*2, CHUNK*4],
    name='input_layer',
    weights_init=blinit.IsotropicGaussian(0.01),
    biases_init=blinit.Constant(0)
)
input_layer_app = input_layer.apply(x)
input_layer.initialize()

middle_layer = brrec.LSTM(
    dim=CHUNK,
    activation=br.Tanh(),
    name='lstm',
    weights_init=blinit.IsotropicGaussian(0.01),
    biases_init=blinit.Constant(0)
)
middle_layer_h, middle_layer_c = middle_layer.apply(input_layer_app)
middle_layer.initialize()

output_layer = br.Linear(
    input_dim=CHUNK,
    output_dim=N_CLASSES,
    name='output_layer',
    weights_init=blinit.IsotropicGaussian(0.01),
    biases_init=blinit.Constant(0)
)
output_layer_app = output_layer.apply(middle_layer_h)
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

algorithm = blalg.GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=blalg.Adam()
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
        blext.ProgressBar()
    ]
)  

main_loop.run()

