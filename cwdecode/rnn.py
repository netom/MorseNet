#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

import theano
import cPickle
import numpy as np
from config import *
import theano.tensor as T
import theano.gradient as G

N_CLASSES  = len(MORSE_CHR)

# Architecture:
#
# 1st layer: sigmoid, (CHUNK,CHUNK)
# 2nd layer: sigmoid, output fed back with new input (CHUNK+N_CLASSES,N_CLASSES)
# 3rd layer: softmax classifier (N_CLASSES, N_CLASSES)
class RNN:
    x   = T.fvector('x') # Input vector
    w1  = T.fmatrix('w1')
    b1  = T.fvector('b1')
    w2  = T.fmatrix('w2')
    b2  = T.fvector('b2')
    o2_ = T.fvector('o2')
    o2  = T.fvector('o2')
    w3  = T.fmatrix('w3')
    b3  = T.fvector('b3')

    o1 = T.nnet.softplus(T.dot(x, w1) + b1)
    o2 = T.nnet.softplus(T.dot(T.concatenate([o1, o2_]), w2) + b2)
    o3 = T.nnet.softmax(T.dot(o2, w3) + b3)

    f = theano.function(inputs=[x, w1, b1, w2, b2, o2_, w3, b3], outputs=[o3, o2])

    def __init__(self):
        self.params = [
            np.random.randn(CHUNK, CHUNK).astype(np.float32) * 0.01,               # w1
            np.zeros(CHUNK, dtype=np.float32),                                     # b1
            np.random.randn(CHUNK+N_CLASSES, N_CLASSES).astype(np.float32) * 0.01, # w2
            np.zeros(N_CLASSES, dtype=np.float32),                                 # b2
            np.zeros(N_CLASSES, dtype=np.float32),                                 # o2
            np.random.randn(N_CLASSES, N_CLASSES).astype(np.float32) * 0.01,       # w3
            np.zeros(N_CLASSES, dtype=np.float32)                                  # b3
        ]

    def propagate(self, x):
        return self.f(*([x] + self.params))

    def neighbour(self, learning_rate):
        nn = Net()
        nn.params = [
            self.params[0] + np.random.randn(CHUNK, CHUNK) * learning_rate,               # w1
            self.params[1] + np.random.randn(CHUNK) * learning_rate,                      # b1
            self.params[2] + np.random.randn(CHUNK+N_CLASSES, N_CLASSES) * learning_rate, # w2
            self.params[3] + np.random.randn(N_CLASSES) * learning_rate,                  # b2
            self.params[4],                                                               # o2
            self.params[5] + np.random.randn(N_CLASSES, N_CLASSES) * learning_rate,       # w3
            self.params[6] + np.random.randn(N_CLASSES) * learning_rate                   # b3
        ]
        return nn

    def mse(self, examples, targets):
        return self.f_mse(*([examples, targets] + self.params))

rnn = RNN()

with open('training_set/sample_0.pickle', 'r') as f:
    chunks, chars = cPickle.load(f)

print "Calculating..."
for chunk in chunks:
    y, o2 = rnn.propagate(chunk)
    rnn.params[4] = o2

print "Done"

