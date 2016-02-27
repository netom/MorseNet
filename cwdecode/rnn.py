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
# 1st layer: softplus, (CHUNK,CHUNK)
# 2nd layer: softplus, output fed back with new input (CHUNK+N_CLASSES,N_CLASSES)
# 3rd layer: softmax classifier (N_CLASSES, N_CLASSES)
class RNN:
    def __init__(self):
        x   = T.fvector('x') # Input vector
        w1  = T.fmatrix('w1')
        b1  = T.fvector('b1')
        w2  = T.fmatrix('w2')
        b2  = T.fvector('b2')
        o2_ = theano.shared(np.zeros(N_CLASSES, dtype=np.float32), 'o2')
        w3  = T.fmatrix('w3')
        b3  = T.fvector('b3')

        o1 = T.nnet.softplus(T.dot(x, w1) + b1)
        o2 = T.nnet.softplus(T.dot(T.concatenate([o1, o2_]), w2) + b2)
        o3 = T.nnet.softmax(T.dot(o2, w3) + b3)

        self.f = theano.function(inputs=[x, w1, b1, w2, b2, w3, b3], outputs=o3, updates=[(o2_, o2)])

        self.params = [
            np.random.randn(CHUNK, CHUNK).astype(np.float32) * 0.01,               # w1
            np.zeros(CHUNK, dtype=np.float32),                                     # b1
            np.random.randn(CHUNK+N_CLASSES, N_CLASSES).astype(np.float32) * 0.01, # w2
            np.zeros(N_CLASSES, dtype=np.float32),                                 # b2
            np.random.randn(N_CLASSES, N_CLASSES).astype(np.float32) * 0.01,       # w3
            np.zeros(N_CLASSES, dtype=np.float32)                                  # b3
        ]

    def propagate(self, x):
        return self.f(*([x] + self.params))

    def neighbour(self, learning_rate):
        params = []
        for i in xrange(len(self.params)):
            params.append(self.params[i] + (np.random.randn(*self.params[i].shape) - 0.5) * learning_rate)
        return params

    def mse(self, example, target):
        return self.f_mse(*([examples, targets] + self.params))

rnn = RNN()

with open('training_set/sample_0.pickle', 'r') as f:
    chunks, chars = cPickle.load(f)

print "Calculating..."
for chunk in chunks:
    y = rnn.propagate(chunk)
    print y

print "Done"

