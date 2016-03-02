#!/usr/bin/env python
#-*- coding: utf-8 -*-

import re
import numpy as np
import scipy.signal as sig
import random
import cPickle

from config import *

def wpm2dit(wpm):
    return 1.2 / wpm

# The length of a dit. Deviation is in percent of dit length
def dit_len(wpm, deviation):
    dl = wpm2dit(wpm)
    return random.normalvariate(dl, dl * deviation)

# The length of a dah
def dah_len(wpm, deviation, dahw = 3.0):
    return dahw * dit_len(wpm, deviation)

# The length of pause between dits and dahs inside a character
def symspace_len(wpm, deviation, symw = 1.0):
    return symw * dit_len(wpm, deviation)

# The length of pause between characters inside a word
def chrspace_len(wpm, deviation, chrw = 3.0):
    return chrw * dit_len(wpm, deviation)

# The length of a space between two words
def wordspace_len(wpm, deviation, wsw = 7.0):
    return wsw * dit_len(wpm, deviation)

# Generates <frames> length of white noise 
def whitenoise(frames, vol):
    return np.random.normal(0, vol, frames)

# Generates <frames> length of popping noise
def impulsenoise(frames, th):
    r = np.random.normal(0.0, 1.0, frames)
    r[r < th] = 0.0
    i = r >= th
    r[r >= th] = 1.0
    #r[i] = (r[i] - th) / (1.0 - th)
    ret = sig.convolve(r, [1.0] * 10 + [-1.0] * 10, mode='same')
    return ret

# Generates a sequence that when multiplied with the signal, it will cause fading
def qsb(frames, vol, f):
    return 1.0 - np.sin(np.linspace(0, 2 * np.pi * frames / FRAMERATE * f, frames)) * vol

# Returns a training sample of (chunks, 1-hot encoded training targets) as a np array
def get_training_data():
    wpm       = random.uniform(5.0, 50)
    deviation = random.uniform(0.0, 0.15)
    wnvol     = random.uniform(0.01, 0.6)
    qsbvol    = random.uniform(0.0, 0.95)
    qsbf      = random.uniform(0.1, 1.0)

    training_data = (np.zeros((SAMPLE_CHUNKS, CHUNK), dtype=np.float32), np.zeros(SAMPLE_CHUNKS, dtype=np.int16))

    padl = int(max(0, random.normalvariate(3, 0.2)) * FRAMERATE) # Padding at the beginning
    pdr  = int(max(0, random.normalvariate(3, 0.2)) * FRAMERATE) # Padding at the end
    sample_i = padl # The actual index in the samlpes
    max_sound_len = 
    el = [];
    c = 0
    prev_c = 0
    while True:
        # Generate a character
        c = get_next_character()
        # ...but not space at the beginning, or repeating spaces
        while prev_c == 0 and c == 0
            c = get_next_character()
        chr_samples = get_samples_for_chr(c)
        # Check if it's too long to fit
        if len(chr_samples) > sample_i + padr:
            break
        # TODO
        if c == ' ':
            el.append((0.0, wordspace_len(wpm, deviation)))
            sound_len += el[-1][1] # This should happend with append
        else:
            for sym in CHARS[c]:
                el.append((1.0, dit_len(wpm, deviation) if sym == '.' else dah_len(wpm, deviation)))
                sound_len += el[-1][1]
                el.append((0.0, symspace_len(wpm, deviation)))
                sound_len += el[-1][1]
            sound_len -= el[-1][1] # This is just ugly
            el = el[:-1]        #
            el.append((0.0, chrspace_len(wpm, deviation)))
            sound_len += el[-1][1]
        timed_chars.append((c, sound_len))
    
    # Generate the sound for it (with padding at the end)
    seql = (sound_len + min(0, random.normalvariate(3, 0.2))) * FRAMERATE
    seql += CHUNK - seql % CHUNK # So seql % CHUNK == 0
    seq = np.zeros(seql, dtype=np.float64)

    i = padl * FRAMERATE
    for e in el:
        l = int(e[1] * FRAMERATE)
        seq[i:(i+l)] = e[0]
        i += l

    return np.reshape(
        (
            seq # On-off sequence
            * np.sin(np.arange(0, len(seq)) * (random.randint(400, 800) * 2 * np.pi / FRAMERATE)) # Baseband signal
            * qsb(len(seq), qsbvol, qsbf)
            + whitenoise(len(seq), wnvol)
            + impulsenoise(len(seq), 4.2)
        ) * 0.25, # Volume
        (seql / CHUNK, CHUNK)
    ).astype(np.float32), timed_chars
    # TODO: filter for clicks (random filter between 1KHz - 50Hz)
    # TODO: QRM

def random_text():
    l = random.randint(5, 10)
    ret = ''.join(random.choice(MORSE_CHR[1:] + [' '] * 6) for _ in xrange(l))
    return re.sub(r'( +)', r' ', ret)

def generate_random_sample(i):
    samplename = 'sample_' + str(i)
    training_data = get_training_data()
    with open(TRAINING_SET_DIR + '/' + samplename + '.pickle', 'w') as f:
        cPickle.dump((chunks, chars), f)

for i in xrange(SETSIZE):
    print i
    generate_random_sample(i)
