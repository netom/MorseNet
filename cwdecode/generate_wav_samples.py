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
import scipy.signal

from config import *

# Spectral inversion for FIR filters
def spectinvert(taps):
    l = len(taps)
    return ([0]*(l/2) + [1] + [0]*(l/2)) - taps

# Returns the dit length in secods from the WPM
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
    # Words per minute
    wpm       = random.uniform(10,  40.0)
    # Error in timing
    deviation = random.uniform(0.0,  0.1)
    # TODO: dit / dah / space weights
    # White noise volime
    wnvol     = random.uniform(0.5,  3.0)
    # QSB volume: 0=no qsb, 1: full silencing QSB
    qsbvol    = random.uniform(0.0,  0.7)
    # QSB frequency in Hertz
    qsbf      = random.uniform(0.1,  0.7)
    # Signal volume
    sigvol    = random.uniform(1.0,  3.3)
    # Signal frequency
    sigf      = random.uniform(500.0, 700.0)
    # Signal phase
    phase     = random.uniform(0.0,  framerate / sigf)
    # Filter lower cutoff
    f1        = random.uniform(sigf - 400, sigf-50)
    # Filter higher cutoff
    f2        = random.uniform(sigf + 50, sigf + 400)
    # Number of taps in the filter
    taps      = 63 # The number of taps of the FIR filter

    audio = np.zeros(seq_length, dtype=np.float64)
    characters = []

    padl = int(max(0, random.normalvariate(1, 0.5)) * framerate) # Padding at the beginning
    i = padl # The actual index in the samlpes
    s_pairs, s_length = get_onoff_data(' ', wpm, deviation)
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
        # Leave some space on the right
        # Always insert a space after the sequence
        if i + length + s_length + 1000 >= seq_length:
            for p in s_pairs:
                audio[i:i+p[1]] = p[0]
                i += p[1]
            characters.append((' ', i / float(framerate)))
            break

        # Write it into the audio data array
        for p in pairs:
            audio[i:i+p[1]] = p[0]
            i += p[1]
        characters.append((c, i / float(framerate)))

    # Set up the bandpass filter
    fil_lowpass = scipy.signal.firwin(taps, f1/(framerate/2))
    fil_highpass = spectinvert(scipy.signal.firwin(taps, f2/(framerate/2)))
    fil_bandreject = fil_lowpass+fil_highpass
    fil_bandpass = spectinvert(fil_bandreject)

    # Remove clicks
    s = sig.convolve(audio, np.array(range(80, 0, -1)) / 3240.0, mode='same') * sigvol
    # Sinewave with phase shift (cw signal)
    s *= np.sin((np.arange(0, seq_length) + phase) * sigf * 2 * np.pi / framerate)
    # QSB
    s *= qsb(seq_length, qsbvol, qsbf)
    # Add white noise
    s += whitenoise(seq_length, wnvol)
    # Add impulse noise
    s += impulsenoise(seq_length, 4.2)
    # Filter signal
    s = scipy.signal.lfilter(fil_bandpass, 1.0, s)
    # AGC with fast attack and slow exponential decay
    a = 0.02  # Attack. The closer to 0 the slower.
    d = 0.002 # Decay. The closer to 0 the slower.
    x = 1.0   # Correction factor 
    for k in xrange(len(s)):
        p = (s[k] / x)**2
        x = max(x + (p - x) * d, x + (p - x) * a)
        s[k] /= x / 1.4142 # I don't know why is this necessary. TODO: figure it out
    #print np.average(s), sum(s**2) / len(s), max(s), min(s) # Debug mean, variance, max, min
    # TODO: QRN
    # Scale and convert to int
    s = (s * 2**12).astype(np.int16)

    return s, characters

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
