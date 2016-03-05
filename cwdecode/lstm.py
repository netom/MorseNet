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
class LSTM:

    def lstm_step(self, x, h_prev, c_prev):
        # Previous output and input concatenated
        h_prevx = T.concatenate([h_prev, x])
        # Forget gate
        f = T.nnet.sigmoid(T.dot(h_prevx, self.wf) + self.bf)
        # Input Gate
        i = T.nnet.sigmoid(T.dot(h_prevx, self.wi) + self.bi)
        # Candidate values for the next state c
        c_ = T.tanh(T.dot(h_prevx, self.wc) + self.bc)
        # Output gate
        o = T.nnet.sigmoid(T.dot(h_prevx, self.wo) + self.bo)

        c = f * c_prev + i * c_
        h = o * T.tanh(c)

        return (h, c)

    def __init__(self, chunks, trgs):

        # The series of input chunks of audio data
        x  = theano.shared(chunks, 'x')
        # 1-hot encoded target characters
        targets = theano.shared(trgs, 'targets')

        #
        # Hyperparameters
        #

        INPUT_SIZE  = CHUNK
        OUTPUT_SIZE = CHUNK

        # Learning rate
        lr = theano.shared(np.float32(0.1))
        lrr = theano.shared(np.float32(0.994))
        lrmin = theano.shared(np.float32(0.0001))

        #
        # Parameters
        #

        # First layer
        w1 = theano.shared((np.random.randn(INPUT_SIZE, INPUT_SIZE) * 0.01).astype(np.float32), 'w1')
        b1 = theano.shared((np.random.randn(INPUT_SIZE) * 0.001).astype(np.float32), 'b1')

        # Second layer is the LSTM

        # Forget gate
        self.wf = theano.shared((np.random.randn(INPUT_SIZE + OUTPUT_SIZE, OUTPUT_SIZE) * 0.01).astype(np.float32))
        self.bf = theano.shared((np.random.randn(OUTPUT_SIZE) * 0.001).astype(np.float32))
        # Input gate
        self.wi = theano.shared((np.random.randn(INPUT_SIZE + OUTPUT_SIZE, OUTPUT_SIZE) * 0.01).astype(np.float32))
        self.bi = theano.shared((np.random.randn(OUTPUT_SIZE) * 0.001).astype(np.float32))
        # Candidate values for next state
        self.wc = theano.shared((np.random.randn(INPUT_SIZE + OUTPUT_SIZE, OUTPUT_SIZE) * 0.01).astype(np.float32))
        self.bc = theano.shared((np.random.randn(OUTPUT_SIZE) * 0.001).astype(np.float32))
        # Output gate
        self.wo = theano.shared((np.random.randn(INPUT_SIZE + OUTPUT_SIZE, OUTPUT_SIZE) * 0.01).astype(np.float32))
        self.bo = theano.shared((np.random.randn(OUTPUT_SIZE) * 0.001).astype(np.float32))

        # Final layer
        w3 = theano.shared((np.random.randn(CHUNK, N_CLASSES) * 0.01).astype(np.float32), 'w3')
        b3 = theano.shared((np.random.randn(N_CLASSES) * 0.001).astype(np.float32), 'b3')

        #
        # Expressions
        #

        # First layer: LRU
        l1 = T.dot(x, w1) + b1
        o1 = T.switch(T.lt(l1, 0), 0, l1)

        # Second layer: LSTM
        o2, _ = theano.scan(
            fn=self.lstm_step,
            outputs_info=(
                np.zeros(OUTPUT_SIZE).astype(np.float32),
                np.zeros(OUTPUT_SIZE).astype(np.float32)
            ),
            sequences=[o1]
        )

        # Third layer: softmax classifier
        o3 = T.nnet.softmax(T.dot(o2[0], w3) + b3)

        self.f = theano.function(
            inputs=[],
            outputs=o3
        )

        loss = T.nnet.categorical_crossentropy(o3, targets).mean() # It also exists as a built-in function :)

        prediction = T.argmax(o3, axis=1)

        errchrs = T.sum(T.switch(T.eq(prediction, targets), 0, 1))

        #norml2reg = 10 * (T.sum(w1**2) + T.sum(w2**2) + T.sum(w3**2))

        self.lossf = theano.function(
            inputs=[],
            outputs=(loss, errchrs)
        )

        # Parameter list
        self.params = [w1, b1, self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo, w3, b3]
        # Gradient list
        grads  = map(lambda p: T.grad(loss, p), self.params)
        # Weight update function
        self.improvef = theano.function(
            inputs  = [],
            outputs = [],
            updates = map(lambda pg: (pg[0], pg[0] - lr * (pg[1] / T.sum(pg[1]**2)**0.5)), zip(self.params, grads)) + [
                (lr, T.maximum(lrmin, lr * lrr))
            ]
        )

    def propagate(self):
        return self.f()

    def loss(self):
        return self.lossf()

    def improve(self):
        self.improvef()

    def get_params(self):
        return map(lambda x: x.get_value(), self.params)

#chunks  = np.zeros((3, SAMPLE_CHUNKS, CHUNK), dtype=np.float32)
#targets = np.zeros((3, SAMPLE_CHUNKS), dtype=np.int64)
#for i in xrange(3):
#    with open('training_set/sample_%d.pickle' % i, 'r') as f:
#        chunks[i], targets[i] = cPickle.load(f)

with open('training_set/sample_0.pickle', 'r') as f:
    chunks, targets = cPickle.load(f)

rnn = LSTM(chunks, targets)

i = 0
loss, errchrs = rnn.loss()
while True:
    prev_loss = loss
    rnn.improve()
    loss, errchrs = rnn.loss()
    i += 1
    if loss < 0.01 and i % 100 == 0:
        print "Saving network..."
        with open('sample_0_overtrain.pickle', 'w') as f:
            cPickle.dump(rnn.get_params(), f)

    print i, errchrs, loss, np.sum(rnn.params[0].get_value()**2), np.sum(rnn.params[2].get_value()**2), np.sum(rnn.params[4].get_value()**2)

    #if prev_loss <= loss:
    #    old_lr = rnn.lr.get_value()
    #    new_lr = np.float32(old_lr * 0.8)
    #    print "Whoops, setting lr: %f -> %f" % (old_lr, new_lr)
    #    rnn.lr.set_value(new_lr)
