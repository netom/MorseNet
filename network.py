#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

import theano
import theano.tensor as T
import theano.gradient as G
#import theano.sandbox.cuda.basic_ops
import numpy as np

# Output layer: EPSILON, A-Z, 0-9, ' ': 37

CHARS = [
   '\0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
    'R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8',
    '9',' '
]

INPUT_SIZE = 10  # Frames / input
N_CLASSES  = len(CHARS)

class LSTM:
    def __init__(self, size):
        # Shared variables

        # State
        self.c  = theano.shared(np.random.randn(INPUT_SIZE), 'c')
        # Output
        self.h  = theano.shared(np.random.randn(INPUT_SIZE), 'h')

        # Forget gate weights and biases
        wf = np.random.randn(size, size * 2)
        bf = np.random.randn(size)
        # Input gate w&b
        wi = np.random.randn(size, size * 2)
        bi = np.random.randn(size)
        # w&b for candidate values for next state
        wc = np.random.randn(size, size * 2)
        bc = np.random.randn(size)
        # Output gate w&c
        wo = np.random.randn(size, size * 2)
        bo = np.random.randn(size)

        # Expressions

        # Input vector
        x  = T.fvector('x') 
        # Previous output and input concatenated
        xh = T.concatenate([x, self.h])
        # Forget gate
        f  = T.nnet.sigmoid(T.dot(wf, xh) + bf)
        # Input Gate
        i  = T.nnet.sigmoid(T.dot(wi, xh) + bi)
        # Candidate values for the next state c
        c_ = T.tanh(T.dot(wc, xh) + bc)
        # Output gate
        o = T.nnet.sigmoid(T.dot(wo, xh) + bo)

        # Forward pass function
        self.function = theano.function([x], [self.h, self.c], updates=[(self.c, f * self.c + i * c_), (self.h, o * T.tanh(self.c))])
        # Jacobian
        self.jacobian = theano.function([x], G.jacobian(o * T.tanh(self.c), x))


def lstm_layer(size, c, h):
    pass

def rlu_layer():
    pass

def softmax_layer():
    pass

lstm = LSTM(INPUT_SIZE)

print "Calculating"
for i in xrange(1):
    print lstm.function([1]*INPUT_SIZE)
    print
    print lstm.jacobian([1]*INPUT_SIZE)

print "Done"

