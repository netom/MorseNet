#!/usr/bin/env python
#-*- coding: utf-8 -*-

# See architecture.md

import theano as Th
import theano.tensor as T
import numpy as np

# Output layer: EPSILON, A-Z, 0-9, ' ': 37

CHARS = [
   '\0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
    'R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8',
    '9',' '
]

INPUT_SIZE = 441  # Frames / input
N_CLASSES  = len(CHARS)

def lstm_layer(size):
    c  = Th.shared(np.random.randn(size), 'c') # State
    h  = Th.shared(np.random.randn(size), 'h') # Output to next layer

    # Affine transform weights biases
    wf = np.random.randn(size, size * 2)
    bf = np.random.randn(size)
    wi = np.random.randn(size, size * 2)
    bi = np.random.randn(size)
    wc = np.random.randn(size, size * 2)
    bc = np.random.randn(size)
    wo = np.random.randn(size, size * 2)
    bo = np.random.randn(size)
 
    x  = T.fvector('x')

    xh = T.concatenate([x, h])

    f  = T.nnet.sigmoid(T.dot(wf, xh) + bf)
    i  = T.nnet.sigmoid(T.dot(wi, xh) + bi)
    c_ = T.tanh(T.dot(wc, xh) + bc)
    o = T.nnet.sigmoid(T.dot(wo, xh) + bo)

    return Th.function([x], [h, c], updates=[(c, f * c + i * c_), (h, o * T.tanh(c))])

def rlu_layer():
    pass

def softmax_layer():
    pass

lstm = lstm_layer(6)

for i in xrange(10):
    print lstm([1,2,3,4,5,6])

