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
        x   = T.fmatrix('x') # Input vectors
        w1  = T.fmatrix('w1')
        b1  = T.fvector('b1')
        w2  = T.fmatrix('w2')
        b2  = T.fvector('b2')
        w3  = T.fmatrix('w3')
        b3  = T.fvector('b3')

        o1, _ = theano.scan(
            fn=lambda x, w1, b1: T.switch(T.dot(x, w1) + b1 < 0, 0, T.dot(x, w1) + b1),
            sequences=[x],
            non_sequences=[w1, b1]
        )

        o2, _ = theano.scan(
            fn=lambda o2_, o1, w2, b2: T.switch(T.dot(T.concatenate([o1, o2_]), w2) + b2 < 0, 0, T.dot(T.concatenate([o1, o2_]), w2) + b2),
            outputs_info=T.zeros_like(b2),
            sequences=[o1],
            non_sequences=[w2, b2]
        )

        o3, _ = theano.scan(
            fn=lambda o2, w3, b3: T.nnet.softmax(T.dot(o2, w3) + b3)[0],
            sequences=[o2],
            non_sequences=[w3, b3]
        )

        self.f = theano.function(
            inputs=[x, w1, b1, w2, b2, w3, b3],
            outputs=o3
        )

        self.params = [
            np.random.randn(CHUNK, CHUNK).astype(np.float32) * 0.001,     # w1
            np.random.randn(CHUNK).astype(np.float32) * 0.001,            # b1
            np.random.randn(CHUNK * 2, CHUNK).astype(np.float32) * 0.001, # w2
            np.random.randn(CHUNK).astype(np.float32) * 0.001,            # b2
            np.random.randn(CHUNK, N_CLASSES).astype(np.float32) * 0.001, # w3
            np.random.randn(N_CLASSES).astype(np.float32) * 0.001         # b3
        ]

    def propagate(self, chunks):
        return self.f(*([chunks] + self.params))

    def neighbour_params(self, learning_rate):
        params = []
        for i in xrange(len(self.params)):
            params.append(self.params[i] + np.random.randn(*self.params[i].shape).astype(np.float32) * learning_rate)
        return params

    def loss(self, chunks, targets):
        predictions = self.propagate(chunks)
        loss = 0
        errchrs = 0
        for p, t in zip(predictions, targets):
            loss += - np.log(p[t])
            if np.argmax(p) != t:
                errchrs += 1
        return (loss, errchrs)
        

rnn = RNN()

with open('training_set/sample_0.pickle', 'r') as f:
    chunks, targets = cPickle.load(f)

i = 0

(loss, errchrs) = rnn.loss(chunks, targets)
while True:
    newparams  = rnn.neighbour_params(0.01 / (1 + i / 100))
    oldparams  = rnn.params
    rnn.params = newparams

    (newloss, newerrchrs) = rnn.loss(chunks, targets)

    if loss <= newloss:
        rnn.params = oldparams
    else:
        loss = newloss
        errchrs = newerrchrs

    i += 1

    print i, errchrs, loss
