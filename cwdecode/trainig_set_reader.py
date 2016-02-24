#!/usr/bin/env python
#-*- coding: utf-8 -*-

from config import *

import wave

for i in xrange(SETSIZE):
    txtname = TRAINING_SET_DIR + '/sample_%d.txt' % i
    wavname = TRAINING_SET_DIR + '/sample_%d.wav' % i

    wr = wave.open(wavname, 'r')

    wlen = wr.getnframes() / FRAMERATE

    # Generate training target
    

    # Generate NN input
