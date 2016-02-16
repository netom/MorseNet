#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math
import theano
import theano.tensor as T
import random
import numpy as np

#    Let s = s0
#    For k = 0 through kmax (exclusive):
#        T ← temperature(k ∕ kmax)
#        Pick a random neighbour, snew ← neighbour(s)
#        If P(E(s), E(snew), T) ≥ random(0, 1), move to the new state:
#           s ← snew
#    Output: the final state s

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def rlu(x):
    return np.maximum(0, x)

class Net:
    # Symbolic parameters
    s_x  = T.fmatrix('x')
    s_y  = T.fmatrix('y') # TODO: this should be a vector
    s_w1 = T.fmatrix('w1')
    s_b1 = T.fvector('b1')
    s_w2 = T.fmatrix('w2')
    s_b2 = T.fvector('b2')

    # Expression that applies the net to a sequence of inputs (batch/minibatch)
    predictions, updates = theano.scan(
        fn=lambda x, w1, b1, w2, b2: T.dot( # This is the neural network itself. TODO: factor this into a factory please.
            T.nnet.sigmoid(T.dot(x, w1) + b1) + 0.1 * x, # Prevent the sigmoid from saturating
            w2
        ) + b2,
        sequences=[s_x],
        non_sequences=[s_w1, s_b1, s_w2, s_b2]
    )

    # A function calculating the predictions
    f = theano.function(inputs=[s_x, s_w1, s_b1, s_w2, s_b2], outputs=predictions)

    # Calculating the mean square error between the predictions for examples and given targets
    f_mse = theano.function(inputs=[s_x, s_y, s_w1, s_b1, s_w2, s_b2], outputs=T.mean(T.pow(predictions - s_y, 2)))

    def __init__(self):
        # Weight and bias matrices
        self.params = [
            np.zeros([2, 2], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros([2, 1], dtype=np.float32),
            np.zeros(1, dtype=np.float32)
        ]

    def propagate(self, x):
        return self.f(*([x] + self.params))

    def randomize(self):
        for i in xrange(len(self.params)):
            self.params[i] = np.random.randn(*self.params[i].shape).astype(np.float32)

    def neighbour(self, learning_rate):
        nn = Net()
        for i in xrange(len(self.params)):
            nn.params[i] = (self.params[i] + (np.random.randn(*self.params[i].shape) - 0.5) * learning_rate).astype(np.float32)
        return nn

    def mse(self, examples, targets):
        return self.f_mse(*([examples, targets] + self.params))

def trans_prob(enew, e):
    return 1.0 if enew < e else 0

examples = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=np.float32)

targets = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=np.float32)

n = Net()
n.randomize()

e = n.mse(examples, targets)
c = 0
while e > 10**(-10) and c < 10**5:
    nn   = n.neighbour(max(e, 0.0001)) # Learning rate dependent on the energy
    enew = nn.mse(examples, targets)

    if trans_prob(enew, e) > random.random():
        e = enew
        n = nn

    if c % 1000 == 0: print e

    c += 1

print
print
print c
print
print n.params
print
print n.propagate(examples)
