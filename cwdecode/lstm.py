#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

import theano
import cPickle
import numpy as np
from config import *
import theano.tensor as T
import theano.gradient as G
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt

N_CLASSES  = len(MORSE_CHR)

# Architecture:
#
# 1st layer: softplus, (CHUNK,CHUNK)
# 2nd layer: softplus, output fed back with new input (CHUNK+N_CLASSES,N_CLASSES)
# 3rd layer: softmax classifier (N_CLASSES, N_CLASSES)
class LSTM:

    def lstm_step(self, x, h_prev, c_prev, wf, bf, wi, bi, wc, bc, wo, bo):
        # Previous output and input concatenated
        h_prevx = T.concatenate([h_prev, x])
        # Forget gate
        f = T.nnet.sigmoid(T.dot(h_prevx, wf) + bf)
        # Input Gate
        i = T.nnet.sigmoid(T.dot(h_prevx, wi) + bi)
        # Candidate values for the next state c
        c_ = T.tanh(T.dot(h_prevx, wc) + bc)
        # Output gate
        o = T.nnet.sigmoid(T.dot(h_prevx, wo) + bo)

        c = f * c_prev + i * c_
        h = o * T.tanh(c)

        return (h, c)

    def __init__(self):

        # The series of input chunks of audio data
        self.x  = theano.shared(np.zeros((SAMPLE_CHUNKS, CHUNK), dtype=np.float32), 'x')
        # 1-hot encoded target characters
        self.targets = theano.shared(np.zeros(SAMPLE_CHUNKS, dtype=np.int64), 'targets')

        #
        # Hyperparameters
        #

        INPUT_SIZE  = CHUNK
        OUTPUT_SIZE = CHUNK

        # Learning rate
        lr = theano.shared(np.float32(0.01))
        lrr = theano.shared(np.float32(0.9965)) # 1/2 per 200
        lrmin = theano.shared(np.float32(0.00001))

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
        w3 = theano.shared((np.random.randn(OUTPUT_SIZE, N_CLASSES) * 0.01).astype(np.float32), 'w3')
        b3 = theano.shared((np.random.randn(N_CLASSES) * 0.001).astype(np.float32), 'b3')

        #
        # Expressions
        #

        # First layer: LRU
        l1 = T.dot(self.x, w1) + b1
        o1 = T.switch(T.lt(l1, 0), 0, l1)

        # Second layer: LSTM
        o2, _ = theano.scan(
            fn=self.lstm_step,
            outputs_info=(
                np.zeros(OUTPUT_SIZE).astype(np.float32),
                np.zeros(OUTPUT_SIZE).astype(np.float32)
            ),
            sequences=[o1],
            non_sequences=[self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo],
            strict=True,
            truncate_gradient=LSTM_TRUNCATED_GRADIENT
        )

        # Third layer: softmax classifier
        o3 = T.nnet.softmax(T.dot(o2[0], w3) + b3)

        self.f = theano.function(
            inputs=[],
            outputs=o3
        )

        loss = T.nnet.categorical_crossentropy(o3, self.targets).mean() # It also exists as a built-in function :)

        prediction = T.argmax(o3, axis=1)

        charposes = T.switch(T.eq(self.targets, 0), 0, 1)
        errchrs = T.sum(T.switch(T.eq(prediction, self.targets), 0, 1))

        #norml2reg = 10 * (T.sum(w1**2) + T.sum(w2**2) + T.sum(w3**2))

        self.lossf = theano.function(
            inputs=[],
            outputs=(loss, errchrs, T.sum(charposes))
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

    def set_task(self, chunks, targets):
        self.x.set_value(chunks)
        self.targets.set_value(targets)

    def try_improve():
        self.try_improvef()

    def propagate(self):
        return self.f()

    def loss(self):
        return self.lossf()

    def improve(self):
        self.improvef()

    def get_params(self):
        return map(lambda x: x.get_value(), self.params)

audio_files   = np.zeros((MINIBATCH_SIZE, SAMPLE_CHUNKS, CHUNK), dtype=np.float32)
target_arrays = np.zeros((MINIBATCH_SIZE, SAMPLE_CHUNKS), dtype=np.int64)
for i in xrange(MINIBATCH_SIZE):
    with open('training_set/sample_%d.pickle' % i, 'r') as f:
        audio_files[i], target_arrays[i] = cPickle.load(f)

#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)

rnn = LSTM()
rnn.set_task(audio_files.reshape((MINIBATCH_SIZE * SAMPLE_CHUNKS, CHUNK)), target_arrays.reshape(MINIBATCH_SIZE * SAMPLE_CHUNKS))

#losses = []
i = 0
while True:

    rnn.improve()
    loss, errchrs, charposes = rnn.loss()

    #losses.append(loss)
    #ax.clear()
    #ax.set_yscale('log')
    #ax.plot(losses)
    #plt.draw()

    if loss < 0.01 and i % 100 == 0:
        print "Saving network..."
        with open('trained_lstm.pickle', 'w') as f:
            cPickle.dump(rnn.get_params(), f)

    print i, errchrs, charposes, loss
    i += 1

plt.show(True)
