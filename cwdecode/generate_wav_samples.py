#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
import scipy.signal as sig
import random
import cPickle
import matplotlib.pyplot as plt
import scipy.io.wavfile

from config import *

def wpm2dit(wpm):
    return 1.2 / wpm

# The length of a dit. Deviation is in percent of dit length
def dit_len(wpm, deviation):
    dl = wpm2dit(wpm)
    return int(random.normalvariate(dl, dl * deviation) * FRAMERATE)

# The length of a dah
def dah_len(wpm, deviation, dahw = 3.0):
    return int(dahw * dit_len(wpm, deviation))

# The length of pause between dits and dahs inside a character
def symspace_len(wpm, deviation, symw = 1.0):
    return int(symw * dit_len(wpm, deviation))

# The length of pause between characters inside a word
def chrspace_len(wpm, deviation, chrw = 3.0):
    return int(chrw * dit_len(wpm, deviation))

# The length of a space between two words
def wordspace_len(wpm, deviation, wsw = 7.0):
    return int(wsw * dit_len(wpm, deviation))

# Generates <frames> length of white noise 
def whitenoise(frames, vol):
    return np.random.normal(0, vol, frames)

# Generates <frames> length of popping noise
def impulsenoise(frames, th):
    r = np.random.normal(0.0, 1.0, frames)
    r[r < th] = 0.0
    i = r >= th
    r[r >= th] = 1.0
    ret = sig.convolve(r, [1.0] * 10 + [-1.0] * 10, mode='same')
    return ret

# Generates a sequence that when multiplied with the signal, it will cause fading
def qsb(frames, vol, f):
    return 1.0 - np.sin(np.linspace(0, 2 * np.pi * frames / FRAMERATE * f, frames)) * vol

# Returns a random morse character
def get_next_character():
    return random.choice(MORSE_CHR[1:] + [' '] * 5)

# Returns: ([(1/0, duration), ...], total length)
def get_onoff_data(c, wpm, deviation, ):
    pairs = []
    length = 0
    if c == ' ':
        pairs.append((0.0, wordspace_len(wpm, deviation)))
        length += pairs[-1][1]
    else:
        last_symspace_len = 0
        for sym in CHARS[c]:
            pairs.append((1.0, dit_len(wpm, deviation) if sym == '.' else dah_len(wpm, deviation)))
            length += pairs[-1][1]
            pairs.append((0.0, symspace_len(wpm, deviation)))
            length += pairs[-1][1]
        length -= pairs[-1][1]
        pairs[-1] = (0.0, (chrspace_len(wpm, deviation)))
        length += pairs[-1][1]
    
    return (pairs, length)

def generate_seq(seq_length, framerate=FRAMERATE, sine=False):
    wpm        = random.uniform(10, 40.0)
    deviation  = random.uniform(0.0, 0.1)
    wnvol      = random.uniform(0.0, 0.5)
    qsbvol     = random.uniform(0.0, 0.7)
    qsbf       = random.uniform(0.1, 0.7)
    sigvol     = random.uniform(0.3, 1.0)

    audio = np.zeros(seq_length, dtype=np.float64)
    characters = []

    padl = int(max(0, random.normalvariate(1, 0.5)) * framerate) # Padding at the beginning
    i = padl # The actual index in the samlpes
    c = ' '
    while True:
        prev_c = c
        c = get_next_character()
        # Generate a character,
        # but not space at the beginning, or repeating spaces
        while prev_c == ' ' and c == ' ':
            c = get_next_character()

        # Get the audio samples for this character
        # TODO: for fuck's sake rename this
        pairs, length = get_onoff_data(c, wpm, deviation)

        # Check if it's too long to fit
        if i + length >= seq_length:
            break

        # Write it into the audio data array
        for p in pairs:
            audio[i:i+p[1]] = p[0]
            i += p[1]
        characters.append((c, i / float(framerate)))

    # TODO: filter for clicks (random filter between 1KHz - 50Hz)
    #
    # Something like this:
    # h=scipy.signal.firwin( numtaps=N, cutoff=40, nyq=Fs/2)
    # y=scipy.signal.lfilter( h, 1.0, x)
    #
    # TODO: QRM
    return ((
        audio * sigvol
        * np.sin(np.arange(0, seq_length) * 625 * 2 * np.pi / framerate, dtype=np.float32) # Baseband signal
        * qsb(seq_length, qsbvol, qsbf)
        + whitenoise(seq_length, wnvol)
        + impulsenoise(seq_length, 4.2)
    ) * 2**13).astype(np.int16), characters

def save_new_batch(i):
    seq_length = int(random.uniform(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH))

    dirname = TRAINING_SET_DIR + '/%04d' % i
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for j in xrange(BATCH_SIZE):
        audio, characters = generate_seq(seq_length)

        scipy.io.wavfile.write(dirname + '/%03d.wav' % j, FRAMERATE, audio)

        with open(dirname + '/%03d.txt' % j, 'w') as f:
            f.write('\n'.join(map(lambda x: x[0] + ',' + str(x[1]), characters)))

        with open(dirname + '/config.txt', 'w') as f:
            f.write('%d' % seq_length)

if not os.path.exists(TRAINING_SET_DIR):
    os.makedirs(TRAINING_SET_DIR)

print "Generating %d batches..." % NUM_BATCHES
for i in xrange(NUM_BATCHES):
    sys.stdout.write("\rGenerating %d... " % i)
    sys.stdout.flush()
    save_new_batch(i)
print "\ndone.\n"
