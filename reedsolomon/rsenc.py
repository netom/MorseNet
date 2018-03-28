#!/usr/bin/env python
#-*- coding: utf8 -*-

import numpy as np

def rs(rows, cols, modulus):
    ret = np.zeros([cols, rows], dtype=np.double)
    for i in xrange(cols):
        for j in xrange(rows):
            ret[i,j] = (j**i) % modulus
    return ret

m = rs(63, 12, 64)

print m
print np.dot([61, 37 ,30 ,28 ,9, 27, 61 ,58, 26, 3, 49, 16], m)
print sum([61, 37 ,30 ,28 ,9, 27, 61 ,58, 26, 3, 49, 16]) % 64
