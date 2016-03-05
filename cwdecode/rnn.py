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
        self.lr = theano.shared(np.float32(0.1), 'lr')
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

        #loss = -T.mean(T.log(o3)[T.arange(targets.shape[0]), targets])
        #loss = -T.mean(T.log(o3[T.arange(targets.shape[0]), targets]))
        loss = T.mean(T.nnet.categorical_crossentropy(o3, targets))

        prediction = T.argmax(o3, axis=1)

        errchrs = T.sum(T.switch(T.eq(prediction, T.argmax(targets)), 0, 1))

        #norml2reg = 10 * (T.sum(w1**2) + T.sum(w2**2) + T.sum(w3**2))

        self.lossf = theano.function(
            inputs=[],
            outputs=(loss, errchrs)
        )

        self.params = [w1, b1, w2, b2, w3, b3]

        # Gradient list
        grads  = map(lambda p: T.grad(loss, p), self.params)
        # Weight update function
        self.improvef = theano.function(
            inputs  = [],
            outputs = [],
            profile = True,
            updates = map(lambda pg: (pg[0], pg[0] - self.lr * (pg[1] / T.sum(pg[1]**2)**0.5)), zip(self.params, grads))
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

rnn = RNN(chunks, targets)

i = 0
loss, errchrs = rnn.loss()
while True:
    #prev_loss = loss
    rnn.improve()
    loss, errchrs = rnn.loss()
    i += 1
    #if loss < 0.01 and i % 100 == 0:
    #    print "Saving network..."
    #    with open('sample_0_overtrain.pickle', 'w') as f:
    #        cPickle.dump(rnn.get_params(), f)

    #print i, errchrs, loss, np.sum(rnn.params[0].get_value()**2), np.sum(rnn.params[2].get_value()**2), np.sum(rnn.params[4].get_value()**2)
    print i, errchrs, loss
    #theano.printing.pydotprint(rnn.improvef, "graph_improvef.png")
    #theano.printing.pydotprint(rnn.lossf, "graph_lossf.png")

    #if prev_loss <= loss:
    #    old_lr = rnn.lr.get_value()
    #    new_lr = np.float32(old_lr * 0.8)
    #    print "Whoops, setting lr: %f -> %f" % (old_lr, new_lr)
    #    rnn.lr.set_value(new_lr)
    if i >= 10:
        break
