#!/usr/bin/env python
#-*- coding: utf-8 -*-

import re
import numpy as np
import scipy.signal as sig
import random
import cPickle
import matplotlib.pyplot as plt

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

# Returns a training sample of (chunks, 1-hot encoded training targets) as a np array
def generate_batch():
    seq_length = int(random.uniform(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH))

    xb = np.zeros((seq_length, BATCH_SIZE, CHUNK), dtype=np.float32)
    yb = np.zeros((seq_length, BATCH_SIZE), dtype=np.int64)

    for i in xrange(BATCH_SIZE):
       x, y = generate_seq(seq_length)
       xb[:,i,:] = x
       yb[:,i]   = y

    return (xb, yb)

def generate_seq(seq_length):
    wpm        = random.uniform(15.0, 25.0)
    deviation  = random.uniform(0.0, 0.1)
    wnvol      = random.uniform(0.0, 0.3)
    qsbvol     = random.uniform(0.0, 0.7)
    qsbf       = random.uniform(0.1, 0.5)

    x = np.zeros(seq_length * CHUNK, dtype=np.float32)
    y = np.zeros(seq_length, dtype=np.int64)

    padl = int(max(0, random.normalvariate(1, 0.2)) * FRAMERATE) # Padding at the beginning
    i = padl # The actual index in the samlpes
    el = [];
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
        if i + length >= seq_length * CHUNK:
            break

        # Write it into the big data array
        for p in pairs:
            x[i:i+p[1]] = p[0]
            i += p[1]
        y[i // CHUNK] = MORSE_ORD[c]

    return ((
        x
        #* np.sin(np.arange(0, len(audio_data)) * (random.randint(500, 700) * 2 * np.pi / FRAMERATE), dtype=np.float32) # Baseband signal
        * qsb(len(x), qsbvol, qsbf)
        + whitenoise(len(x), wnvol)
        #+ impulsenoise(len(audio_data), 4.2)
    ) * 0.25).reshape((seq_length, CHUNK)).astype(np.float32), y
    # TODO: filter for clicks (random filter between 1KHz - 50Hz)
    # TODO: QRM

def save_new_batch(i):
    batch = generate_batch()
    #plt.plot(training_data[0].reshape(CHUNK * SAMPLE_CHUNKS))
    #plt.show()
    with open(TRAINING_SET_DIR + '/batch_' + str(i) + '.pickle', 'w') as f:
        cPickle.dump(batch, f)

for i in xrange(NUM_BATCHES):
    print i
    save_new_batch(i)

