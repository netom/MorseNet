#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

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
    for i in xrange(num_batches):
        with open('training_set/batch_%d.pickle' % (offset + i), 'r') as f:
            x_b, y_b = cPickle.load(f)
        x.append(x_b)
        y.append(y_b)

    return DataStream(dataset=IterableDataset({'x': x, 'y': y}))

stream_train = get_datastream(20, 0)
stream_test  = get_datastream(20, 20)

x = T.ftensor3('x')
y = T.lmatrix('y')

input_layer = br.MLP(
    activations=[br.Rectifier()],
    dims=[CHUNK, CHUNK*4],
    name='input_layer',
    weights_init=blinit.IsotropicGaussian(),
    biases_init=blinit.Constant(0)
)
input_layer_app = input_layer.apply(x)

middle_layer = brrec.LSTM(
    dim=CHUNK,
    activation=br.Tanh(),
    name='lstm',
    weights_init=blinit.IsotropicGaussian(),
    biases_init=blinit.Constant(0)
)
middle_layer_h, middle_layer_c = middle_layer.apply(input_layer_app)

output_layer = br.MLP(
    activations=[br.Softmax()],
    dims=[CHUNK, N_CLASSES],
    name='output_layer',
    weights_init=blinit.IsotropicGaussian(),
    biases_init=blinit.Constant(0)
)

y_hat = output_layer.apply(middle_layer_h[-1])

cost = brcost.CategoricalCrossEntropy().apply(y.flatten(), y_hat)
cost.name = 'cost'

input_layer.initialize()
middle_layer.initialize()
output_layer.initialize()

cg = blgraph.ComputationGraph(cost)

algorithm = blalg.GradientDescent(
    cost=cost,
    parameters=cg.parameters,
    step_rule=blalg.Adam()
)

test_monitor = blmon.DataStreamMonitoring(
    variables=[cost],
    data_stream=stream_test,
    prefix='test'
)

train_monitor = blmon.TrainingDataMonitoring(
    variables=[cost],
    prefix='train',
    after_epoch=True
)

main_loop = blml.MainLoop(algorithm, stream_train,
    extensions=[
        test_monitor,
        train_monitor,
        blext.FinishAfter(after_n_epochs=10),
        blext.Printing(),
        blext.ProgressBar()
    ]
)

main_loop.run()

