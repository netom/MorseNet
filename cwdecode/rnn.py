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

    def __init__(self, chunks, trgs):
        x  = theano.shared(chunks, 'x')
        w1 = theano.shared((np.random.randn(CHUNK, CHUNK) * 0.01).astype(np.float32), 'w1')
        b1 = theano.shared((np.random.randn(CHUNK) * 0.001).astype(np.float32), 'b1')
        w2 = theano.shared((np.random.randn(CHUNK * 2, CHUNK) * 0.01).astype(np.float32), 'w2')
        b2 = theano.shared((np.random.randn(CHUNK) * 0.001).astype(np.float32), 'b2')
        w3 = theano.shared((np.random.randn(CHUNK, N_CLASSES) * 0.01).astype(np.float32), 'w3')
        b3 = theano.shared((np.random.randn(N_CLASSES) * 0.001).astype(np.float32), 'b3')
        lr = theano.shared(np.float32(0.001))
        targets = theano.shared(trgs, 'targets')

        l1 = T.dot(x, w1) + b1
        o1 = T.switch(T.lt(l1, 0), 0, l1)

        o2, _ = theano.scan(
            fn=lambda o1, o2_: T.switch(T.dot(T.concatenate([o1, o2_]), w2) + b2 < 0, 0, T.dot(T.concatenate([o1, o2_]), w2) + b2),
            outputs_info=T.zeros_like(b2),
            sequences=[o1]
        )

        o3 = T.nnet.softmax(T.dot(o2, w3) + b3)

        self.f = theano.function(
            inputs=[],
            outputs=o3
        )

        #loss = T.sum(- T.log(o3[T.arange(targets.shape[0]), targets])) # TODO: it really does this? :D
        loss = T.nnet.categorical_crossentropy(o3, targets).mean() # It also exists as a built-in function :)

        prediction = T.argmax(o3, axis=1)

        errchrs = T.sum(T.switch(T.eq(prediction, targets), 0, 1))

        #norml2reg = 10 * (T.sum(w1**2) + T.sum(w2**2) + T.sum(w3**2))

        self.lossf = theano.function(
            inputs=[],
            outputs=(loss, errchrs)
        )

        self.params = [w1, b1, w2, b2, w3, b3]

        self.improvef = theano.function(
            inputs=[],
            outputs=[],
            updates=[
                (w1, w1 - lr * T.grad(loss, w1)),
                (b1, b1 - lr * T.grad(loss, b1)),
                (w2, w2 - lr * T.grad(loss, w2)),
                (b2, b2 - lr * T.grad(loss, b2)),
                (w3, w3 - lr * T.grad(loss, w3)),
                (b3, b3 - lr * T.grad(loss, b3))
            ]
        )

    def propagate(self):
        return self.f()

    def loss(self):
        return self.lossf()

    def get_params(self):
        return map(lambda x: x.get_value(), self.params)

    def set_params(self, params):
        for i in xrange(len(params)):
            self.params[i].set_value(params[i])

    def neighbour_params(self, learning_rate):
        params = []
        for param in self.get_params():
            params.append(param + np.random.randn(*param.shape).astype(np.float32) * learning_rate)
        return params

    def improve(self):
        return self.improvef()


with open('training_set/sample_1.pickle', 'r') as f:
    chunks, targets = cPickle.load(f)

rnn = RNN(chunks.astype(np.float32), np.array(targets))

i = 0
while True:
    rnn.improve()
    loss, errchrs = rnn.loss()
    i += 1
    print i, errchrs, loss, np.sum(rnn.params[0].get_value()**2), np.sum(rnn.params[2].get_value()**2), np.sum(rnn.params[4].get_value()**2)
