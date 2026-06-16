#!/usr/bin/env python3

import multiprocessing
import numpy as np
import queue
import random
import scipy.signal as sig
import tensorflow as tf
import threading

from config import *
from numba import njit

if SAMPLE_GENERATOR_WORKER_TYPE == 'process':
    Worker = multiprocessing.Process
    Queue  = multiprocessing.Queue
elif SAMPLE_GENERATOR_WORKER_TYPE == 'thread':
    Worker = threading.Thread
    Queue  = queue.Queue
else:
    raise Exception(f"")

# Spectral inversion for FIR filters
@njit
def spectinvert(taps):
    new_taps = -taps
    new_taps[len(taps) // 2 + 1] += 1.0
    return new_taps

# Returns the dit length in secods from the WPM
@njit
def wpm2dit(wpm):
    return 1.2 / wpm

# The length of a dit. Deviation is in percent of dit length
@njit
def dit_len(wpm, deviation):
    dl = wpm2dit(wpm)
    return int(random.normalvariate(dl, dl * deviation) * FRAMERATE)

# The length of a dah
@njit
def dah_len(wpm, deviation, dahw = 3.0):
    return int(dahw * dit_len(wpm, deviation))

# The length of pause between dits and dahs inside a character
@njit
def symspace_len(wpm, deviation, symw = 1.0):
    return int(symw * dit_len(wpm, deviation))

# The length of pause between characters inside a word
@njit
def chrspace_len(wpm, deviation, chrw = 3.0):
    return int(chrw * dit_len(wpm, deviation))

# The length of a space between two words
@njit
def wordspace_len(wpm, deviation, wsw = 6.0):
    return int(wsw * dit_len(wpm, deviation))

# Generates <frames> length of white noise 
@njit
def whitenoise(frames, vol):
    return np.random.normal(0, vol, frames)

# Generates <frames> length of popping noise
@njit
def impulsenoise(frames, th):
    r = np.asarray(np.random.normal(0.0, 1.0, frames), dtype=np.float32)
    r[r < th] = 0.0
    i = r >= th
    r[r >= th] = 1.0

    kernel = np.ones((20,), dtype=np.float32)
    kernel[10:] = -1.0

    ret = np.convolve(r, kernel, mode='same')
    return ret

# Generates a sequence that when multiplied with the signal, it will cause fading
@njit
def qsb(frames, vol, f):
    return 1.0 - np.sin(np.linspace(0, 2 * np.pi * frames / FRAMERATE * f, frames)) * vol

@njit
def process_audio_before_filter(audio, sigvol, seq_length, phase, sigf, framerate, qsbvol, qsbf, wnvol):
    # Remove clicks
    s = np.convolve(
        audio,
        np.arange(80.0, 0.0, -1.0, dtype=np.float32) / 3240.0,
        mode='same'
    ).astype(np.float32)
    # Adjust volume
    s *= sigvol
    # Sinewave with phase shift (cw signal)
    s *= np.sin((np.arange(0, seq_length) + phase) * sigf * 2 * np.pi / framerate)
    # QSB
    s *= qsb(seq_length, qsbvol, qsbf)
    # Add white noise
    s += whitenoise(seq_length, wnvol)
    # Add impulse noise
    s += impulsenoise(seq_length, 4.2)

    return s

@njit
def process_audio_after_filter(audio, normalize=False):
    s = audio
    # AGC with fast attack and slow exponential decay
    #a = 0.02  # Attack. The closer to 0 the slower.
    #d = 0.002 # Decay. The closer to 0 the slower.
    #agc_coeff = 1.0   # Correction factor 
    #for k in range(len(s)):
    #    s[k] *= agc_coeff
    #    err = s[k]**2 - 1.0
    #    if err >= 0:
    #        # Level is too high
    #        agc_coeff -= abs(err * a)
    #    else:
    #        # Level is too low
    #        agc_coeff += abs(err * d)
    #s *= 1.56

    s /= np.sqrt(np.average(s**2))

    if normalize:
        s = (s - np.float32(np.mean(s))) / np.float32(np.std(s))

    s = (s * np.float32(2**12)).astype(np.int16)

    return s

# Returns a random morse character
def get_next_character():
    return random.choice(MORSE_CHR[1:] + [' '] * 5)

# Returns: ([(1/0, duration), ...], total length)
def morse_marks(c, wpm, deviation):
    marks = []
    length = 0
    if c == ' ':
        marks.append((0.0, wordspace_len(wpm, deviation)))
        length += marks[-1][1]
    else:
        last_symspace_len = 0
        for sym in CHARS[c]:
            marks.append((1.0, dit_len(wpm, deviation) if sym == '.' else dah_len(wpm, deviation)))
            length += marks[-1][1]
            marks.append((0.0, symspace_len(wpm, deviation)))
            length += marks[-1][1]
        length -= marks[-1][1]
        marks[-1] = (0.0, (chrspace_len(wpm, deviation)))
        length += marks[-1][1]
    
    return (marks, length)

def generate_seq(seq_length, framerate=FRAMERATE, normalize=False):
    # Words per minute
    wpm       = random.uniform(WPM_MIN,  WPM_MAX)
    # Error in timing
    deviation = np.float32(random.uniform(0.0,  0.2))
    # White noise volume
    wnvol     = np.float32(random.uniform(0.3,  4.0))
    # QSB volume: 0=no qsb, 1: full silencing QSB
    qsbvol    = np.float32(random.uniform(0.0,  0.7))
    # QSB frequency in Hertz
    qsbf      = np.float32(random.uniform(0.1,  0.7))
    # Signal volume
    sigvol    = np.float32(random.uniform(1.0,  4.0))
    # Signal frequency
    sigf      = np.float32(random.uniform(500.0, 700.0))
    # Signal phase
    phase     = np.float32(random.uniform(0.0,  framerate / sigf))
    # Filter lower cutoff
    f1        = np.float32(random.uniform(sigf - np.float32(400.0), sigf - np.float32(100.0)))
    # Filter higher cutoff
    f2        = np.float32(random.uniform(sigf + np.float32(400.0), sigf + np.float32(100.0)))
    # Number of taps in the filter
    taps      = 63 # The number of taps of the FIR filter

    audio = np.zeros(seq_length, dtype=np.float32)
    characters = []

    padl = int(max(0, random.normalvariate(1, 0.5)) * framerate) # Padding at the beginning
    padr = int(max(0, random.normalvariate(1, 0.5)) * framerate) # Padding at the end

    i      = padl # Current audio sample index. Start at padl.
    prev_c = ' ' # Previous character
    c      = ' '  # Current character. Hack: prevent starting with space.

    while len(characters) < SEQ_MAX_CHARS:
        prev_c = c
        c = get_next_character()

        # Avoid runs of spaces. Also prevents starting with a space (see above).
        if prev_c == ' ':
            while c == ' ':
                c = get_next_character()

        # Get the audio samples for this character
        pairs, length = morse_marks(c, wpm, deviation)

        # Check if it's too long to fit
        if i + length + padr >= seq_length:
            break

        # Write it into the audio data array
        for p in pairs:
            audio[i:i+p[1]] = p[0]
            i += p[1]

        characters.append(c)

    # If the last character is a space, just remove it.
    while characters[-1] == ' ':
        characters.pop()

    # Set up the bandpass filter
    fil_lowpass = np.asarray(sig.firwin(taps, f1/(framerate/2)), dtype=np.float32)
    fil_highpass = spectinvert(np.asarray(sig.firwin(taps, f2/(framerate/2)), dtype=np.float32))
    fil_bandreject = fil_lowpass + fil_highpass
    fil_bandpass = spectinvert(fil_bandreject)

    s = process_audio_before_filter(audio, sigvol, seq_length, phase, sigf, framerate, qsbvol, qsbf, wnvol)
    s = sig.lfilter(fil_bandpass, np.float32(1.0), s)
    s = process_audio_after_filter(s, normalize=normalize)

    return s, ''.join(characters)

# List of worker processes or threads
workers = []

work_queue = Queue(BATCH_SIZE * 2)

def dowork():
    global work_queue

    while True:
        audio, characters = generate_seq(SEQ_LENGTH_FRAMES, FRAMERATE, normalize=True)

        audio = audio.astype(np.float32)
        audio = np.reshape(audio,  (SEQ_LENGTH_FRAMES // CHUNK, CHUNK))

        indices = np.asarray(range(len(characters)), dtype=np.int64)
        indices = np.reshape(indices, (len(indices),1))

        values = np.asarray([MORSE_ORD[c] for c in characters], dtype=np.int32)
        dense_shape = np.asarray([SEQ_MAX_CHARS], dtype=np.int64)

        work_queue.put((audio, indices, values, dense_shape))

def start_workers():
    global workers

    if len(workers) > 0:
        return

    num_workers = SAMPLE_GENERATOR_WORKERS

    if num_workers < 0:
        raise Exception(f'Number of workers must be greater than 0, got {num_workers}')

    if type(num_workers) == float:
        num_workers = round(multiprocessing.cpu_count() * num_workers)

    for i in range(num_workers):
        w = Worker(target=dowork)
        w.daemon=True
        workers.append(w)
        w.start()

# A generator yielding an audio array, and a sparse tensor of lablels for CTC
# functions
def seq_generator():
    start_workers()
    while True:
        audio, indices, values, dense_shape = work_queue.get()
        sparse_label = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=np.asarray((40,), dtype=np.int64) #dense_shape
        )
        yield audio, sparse_label

def save_files(dirname, seq_length, batch_size):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range(batch_size):
        w = 20
        n = w*i//batch_size
        sys.stdout.write("\r[%s>%s] %4d/%4d  " % ("="*n, " "*(w-n), i, batch_size))
        sys.stdout.flush()
        filename = dirname + '/%03d.wav' % i

        audio, characters = generate_seq(seq_length, FRAMERATE, normalize=True)

        # scale and convert to int
        audio = (audio * 2**12).astype(np.int16)

        scipy.io.wavfile.write(filename, FRAMERATE, audio)

        with open(dirname + '/%03d.txt' % i, 'w') as f:
            f.write(characters)

        with open(dirname + '/config.txt', 'w') as f:
            f.write('%d' % seq_length)
    print("")

if __name__ == "__main__":
    import os
    import sys
    import argparse
    import scipy.io.wavfile

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'dirname', metavar='DIRNAME', type=str,
        help='name of the directory into wich the output .wav files are to be saved'
    )
    parser.add_argument(
        'batchsize', metavar='BATCHSIZE', type=int,
        help='the number of examples to generate'
    )
    parser.add_argument(
        '--length', metavar='LENGTH', type=int, default=SEQ_LENGTH_SECONDS,
        help='the approximate length of the samples in whole seconds'
    )

    args = parser.parse_args()

    save_files(args.dirname, args.length * FRAMERATE, args.batchsize)

