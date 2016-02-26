#!/usr/bin/env python
#-*- coding: utf-8 -*-

import time
import cPickle
import pyaudio
import numpy as np
from config import *

p = pyaudio.PyAudio()

stream = p.open(
    output_device_index = DEVICE,
    format = pyaudio.paFloat32,
    channels = 1,
    rate = FRAMERATE,
    input = False,
    output = True,
    frames_per_buffer = CHUNK
)

for i in xrange(SETSIZE):
    print "Playing %d..." % i

    fname = TRAINING_SET_DIR + '/sample_%d.pickle' % i

    with open(fname, 'r') as f:
        chunks, chars = cPickle.load(f)

    for chunk, char in zip(chunks, chars):
        stream.write(chunk.tostring(), CHUNK)
        if char != 0:
            print MORSE_CHR[char]

time.sleep(CHUNK/float(FRAMERATE))

stream.stop_stream()
stream.close()
