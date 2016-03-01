#!/usr/bin/env python
#-*- coding: utf-8 -*-

import re
import numpy as np
import scipy.signal as sig
import random
import cPickle

from config import *

def random_text_length_min():
    return 5

def random_text_length_max():
    return 100

def wpm2dit(wpm):
    return 1.2 / wpm

# Deviation is in percent of dit length
def dit_len(wpm, deviation):
    dl = wpm2dit(wpm)
    return random.normalvariate(dl, dl * deviation) # dit length

def dah_len(wpm, deviation, dahw = 3.0):
    return dahw * dit_len(wpm, deviation)

def symspace_len(wpm, deviation, symw = 1.0):
    return symw * dit_len(wpm, deviation)

def chrspace_len(wpm, deviation, chrw = 3.0):
    return chrw * dit_len(wpm, deviation)

def wordspace_len(wpm, deviation, wsw = 7.0):
    return wsw * dit_len(wpm, deviation)

def whitenoise(frames, vol):
    return np.random.normal(0, vol, frames)

def impulsenoise(frames, th):
    r = np.random.normal(0.0, 1.0, frames)
    r[r < th] = 0.0
    i = r >= th
    r[r >= th] = 1.0
    #r[i] = (r[i] - th) / (1.0 - th)
    ret = sig.convolve(r, [1.0] * 10 + [-1.0] * 10, mode='same')
    return ret

def qsb(frames, vol, f):
    return 1.0 - np.sin(np.linspace(0, 2 * np.pi * frames / FRAMERATE * f, frames)) * vol

def txt2morse(txt):
    wpm       = random.uniform(5.0, 50)
    deviation = random.uniform(0.0, 0.15)
    wnvol     = random.uniform(0.01, 0.6)
    qsbvol    = random.uniform(0.0, 0.95)
    qsbf      = random.uniform(0.1, 1.0)

    timed_chars = []

    # Build the event list
    padh = max(0, random.normalvariate(3, 0.2)) # Padding at the beginning
    soundl = padh # Sound length
    el = [];
    for c in txt:
        if c == ' ':
            el.append((0.0, wordspace_len(wpm, deviation)))
            soundl += el[-1][1] # This should happend with append
        else:
            for sym in CHARS[c]:
                el.append((1.0, dit_len(wpm, deviation) if sym == '.' else dah_len(wpm, deviation)))
                soundl += el[-1][1]
                el.append((0.0, symspace_len(wpm, deviation)))
                soundl += el[-1][1]
            soundl -= el[-1][1] # This is just ugly
            el = el[:-1]        #
            el.append((0.0, chrspace_len(wpm, deviation)))
            soundl += el[-1][1]
        timed_chars.append((c, soundl))
    
    # Generate the sound for it (with padding at the end)
    seql = (soundl + min(0, random.normalvariate(3, 0.2))) * FRAMERATE
    seql += CHUNK - seql % CHUNK # So seql % CHUNK == 0
    seq = np.zeros(seql, dtype=np.float64)

    i = padh * FRAMERATE
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
    # TODO: filter for clicks (random filter?)
    # TODO: QRM

def random_text():
    l = random.randint(5, 10)
    ret = ''.join(random.choice(MORSE_CHR[1:] + [' '] * 6) for _ in xrange(l))
    return re.sub(r'( +)', r' ', ret)

def generate_random_sample(i):
    samplename = 'sample_' + str(i)
    txt = random_text()

    # Generate training target and RNN input
    chunks, timed_chars = txt2morse(txt)
    print ''.join(map(lambda x: x[0], timed_chars))

    # Loop through de wave data and take CHUNK samples at a time.
    # Advance a time counter. If the time just have past a character,
    # put that character into the training target and remove from the list.
    # Put the lists into a tuple and pickle the whole thing

    chars = []
    t = 0
    for i in xrange(len(chunks)):
        t += CHUNK_T

        if timed_chars[0][1] <= t:
            char = MORSE_ORD[timed_chars[0][0]]
            timed_chars = timed_chars[1:]
        else:
            char = 0

        chars.append(char)

        if len(timed_chars) <= 0:
            break;

    with open(TRAINING_SET_DIR + '/' + samplename + '.pickle', 'w') as f:
        cPickle.dump((chunks, chars), f)

for i in xrange(SETSIZE):
    print i
    generate_random_sample(i)
