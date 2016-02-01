#!/usr/bin/env python
#-*- coding: utf-8 -*-

# The problem is a transcription problem

import wave
import numpy as np
import scipy.signal as sig
import random

SETSIZE = 10000
TRAINING_SET_DIR = 'training_set'
CHARS = {
    'A': '.-',
    'B': '-...',
    'C': '-.-.',
    'D': '-...',
    'E': '.',
    'F': '..-.',
    'G': '--.',
    'H': '....',
    'I': '..',
    'J': '.---',
    'K': '-.-',
    'L': '.-..',
    'M': '--',
    'N': '-.',
    'O': '---',
    'P': '.--.',
    'Q': '--.-',
    'R': '.-.',
    'S': '...',
    'T': '-',
    'U': '..-',
    'V': '...-',
    'W': '.--',
    'X': '-..-',
    'Y': '-.--',
    'Z': '--..',
    '0': '-----',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    ' ': None
}

def random_text_length_min():
    return 5

def random_text_length_max():
    return 100

def framerate():
    return 44100.0

def wpm2dit(wpm):
    return 1.2 / wpm

# Deviation is normalized
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
    return 1.0 - np.sin(np.linspace(0, 2 * np.pi * frames / framerate() * f, frames)) * vol

def txt2morse(txt):
    wpm       = random.uniform(5.0, 50)
    deviation = random.uniform(0.0, 0.15)
    wnvol     = random.uniform(0.01, 0.6)
    qsbvol    = random.uniform(0.0, 0.95)
    qsbf      = random.uniform(0.1, 1.0)

    timed_txt = ''

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
        timed_txt += '%s,%f\n' % (c, soundl)
    
    # Generate the sound for it (with padding at the end)
    seq = np.zeros((soundl + min(0, random.normalvariate(3, 0.2))) * framerate(), dtype=np.float64)

    i = padh * framerate()
    for e in el:
        l = int(e[1] * framerate())
        seq[i:(i+l)] = e[0]
        i += l

    return (
        (
            seq # On-off sequence
            * np.sin(np.arange(0, len(seq)) * (random.randint(400, 800) * 2 * np.pi / framerate())) # Baseband signal
            * qsb(len(seq), qsbvol, qsbf)
            + whitenoise(len(seq), wnvol)
            + impulsenoise(len(seq), 4.2)
        ) * (2 ** 13)
    ).astype(np.int16), timed_txt
    # TODO: filter for clicks (random filter?)
    # TODO: QRM

def random_text():
    ret = ''
    l = random.randint(5, 100)
    return ''.join(random.choice(CHARS.keys()) for _ in xrange(l))

def generate_random_sample(i):
    #samplename = ''.join(random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']) for _ in xrange(20))
    samplename = 'sample_' + str(i)
    txt = random_text()
    # TODO: timed training output
    data, timed_txt = txt2morse(txt)

    with open(TRAINING_SET_DIR + '/' + samplename + '.txt', 'w') as f:
        f.write(timed_txt)

    w = wave.open(TRAINING_SET_DIR + '/' + samplename + '.wav', 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(int(framerate()))
    w.writeframes(data)

for i in xrange(SETSIZE):
    print i
    generate_random_sample(i)
