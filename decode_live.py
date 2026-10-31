#!/usr/bin/env python3

"""Decode new microphone audio using persistent LSTM and CTC state."""

import argparse
import queue
import sys
import time
from math import ceil, gcd
from threading import Event

import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly

from model import build_model

from config.audio import *
from config.live import *
from config.morse import *
from config.train import *

class StreamingResampler:
    """Polyphase resampling with aligned history and filter lookahead."""

    def __init__(self, source_rate):
        divisor = gcd(source_rate, FRAMERATE)
        self.up = FRAMERATE // divisor
        self.down = source_rate // divisor
        self.context = ceil(10 * max(self.up, self.down) / self.up / self.down) * self.down
        self.history = np.zeros(self.context, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)

    def process(self, audio):
        if self.up == self.down:
            return np.asarray(audio, dtype=np.float32)
        self.pending = np.concatenate((self.pending, audio))
        count = (len(self.pending) - self.context) // self.down * self.down
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        window = np.concatenate((self.history, self.pending[:count + self.context]))
        resampled = resample_poly(window, self.up, self.down)
        offset = self.context * self.up // self.down
        output = resampled[offset:offset + count * self.up // self.down]
        self.history = np.concatenate((self.history, self.pending[:count]))[-self.context:]
        self.pending = self.pending[count:]
        return output


class StatefulInference:
    """Reuse trained layers with explicit hidden/cell state for each LSTM."""

    def __init__(self, model):
        self.layers = []
        self.states = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                settings = layer.get_config()
                settings.update(return_state=True, stateful=False)
                recurrent = tf.keras.layers.LSTM.from_config(settings)
                recurrent.build((1, None, int(layer.cell.kernel.shape[0])))
                recurrent.set_weights(layer.get_weights())
                self.layers.append(recurrent)
                self.states.append((
                    tf.zeros((1, layer.units)), tf.zeros((1, layer.units))
                ))
            else:
                self.layers.append(layer)
        self.previous_class = 0
        self.forward.get_concrete_function(
            tf.TensorSpec((1, None, CHUNK), tf.float32),
            tuple((tf.TensorSpec(hidden.shape, hidden.dtype),
                   tf.TensorSpec(cell.shape, cell.dtype))
                  for hidden, cell in self.states)
        )

    def reset(self):
        self.states = tuple(
            (tf.zeros_like(hidden), tf.zeros_like(cell))
            for hidden, cell in self.states
        )
        self.previous_class = 0

    @tf.function(reduce_retracing=True)
    def forward(self, audio, states):
        next_states = []
        state_index = 0
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                audio, hidden, cell = layer(
                    audio, initial_state=states[state_index], training=False
                )
                next_states.append((hidden, cell))
                state_index += 1
            else:
                audio = layer(audio, training=False)
        return audio, tuple(next_states)

    def process(self, frames):
        logits, self.states = self.forward(
            tf.convert_to_tensor(frames[None], dtype=tf.float32), tuple(self.states)
        )
        return logits[0].numpy()

    def decode(self, logits):
        text = []
        for character_class in np.argmax(logits, axis=-1):
            character_class = int(character_class)
            if character_class and character_class != self.previous_class:
                text.append(MORSE_CHR[character_class])
            self.previous_class = character_class
        return ''.join(text)


class StreamingDecoder:
    """Frame new resampled audio and normalize it with causal running moments."""

    def __init__(self, model, source_rate):
        self.inference = StatefulInference(model)
        self.resampler = StreamingResampler(source_rate)
        self.pending = np.empty(0, dtype=np.float32)
        self.mean = None
        self.second_moment = None
        if LIVE_NORMALIZATION_SECONDS <= 0:
            raise ValueError("LIVE_NORMALIZATION_SECONDS must be positive")
        self.decay = np.exp(-CHUNK / (FRAMERATE * LIVE_NORMALIZATION_SECONDS))
        self.source_rate = source_rate

    def reset(self):
        self.inference.reset()
        self.resampler = StreamingResampler(self.source_rate)
        self.pending = np.empty(0, dtype=np.float32)
        self.mean = None
        self.second_moment = None

    def process(self, audio):
        self.pending = np.concatenate((self.pending, self.resampler.process(audio)))
        count = len(self.pending) // CHUNK * CHUNK
        if not count:
            return ''
        frames = self.pending[:count].reshape(-1, CHUNK)
        self.pending = self.pending[count:]
        normalized = np.empty_like(frames)
        for index, frame in enumerate(frames):
            mean = float(np.mean(frame))
            second_moment = float(np.mean(frame ** 2))
            if self.mean is None:
                self.mean, self.second_moment = mean, second_moment
            else:
                self.mean = self.decay * self.mean + (1 - self.decay) * mean
                self.second_moment = (
                    self.decay * self.second_moment + (1 - self.decay) * second_moment
                )
            scale = np.sqrt(max(self.second_moment - self.mean ** 2, np.finfo(np.float32).eps))
            normalized[index] = (frame - self.mean) / scale
        logits = self.inference.process(normalized)
        # Exact digital silence should not emit hallucinated characters.
        silent = np.all(frames == 0, axis=1)
        logits[silent] = 0
        return self.inference.decode(logits)


class AudioCapture:
    """Keep the audio callback independent of inference and flag missing audio."""

    def __init__(self, capacity, continue_flag, overflow_flag=2):
        self.blocks = queue.Queue(maxsize=capacity)
        self.dropped = Event()
        self.continue_flag = continue_flag
        self.overflow_flag = overflow_flag
        self.pending_gap = ''

    def callback(self, in_data, frame_count, time_info, status):
        try:
            self.blocks.put_nowait((in_data, status & self.overflow_flag))
        except queue.Full:
            self.dropped.set()
        return None, self.continue_flag

    def read(self):
        if self.dropped.is_set():
            self.pending_gap = 'capture queue overflow (inference fell behind)'
            self.dropped.clear()
            # Discard stale blocks rather than replaying an ever-growing delay.
            while True:
                try:
                    self.blocks.get_nowait()
                except queue.Empty:
                    break
        data, status = self.blocks.get(timeout=1.0)
        reason = self.pending_gap
        self.pending_gap = ''
        if status:
            reason = 'PortAudio input overflow'
        blocks = [data]
        # Process queued audio together, without retaining pre-gap samples.
        for _ in range(self.blocks.qsize()):
            try:
                data, status = self.blocks.get_nowait()
            except queue.Empty:
                break
            if status:
                blocks = []
                reason = 'PortAudio input overflow'
            blocks.append(data)
        return np.frombuffer(b''.join(blocks), dtype=np.int16).astype(np.float32), reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_DIR)
    parser.add_argument('--device', type=int, default=DEVICE, help='Input device index')
    parser.add_argument('--list-devices', action='store_true', help='List input devices and exit')
    args = parser.parse_args()
    if LIVE_STATEFUL_CPU_THREADS <= 0:
        parser.error("LIVE_STATEFUL_CPU_THREADS must be positive")
    tf.config.threading.set_intra_op_parallelism_threads(LIVE_STATEFUL_CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(LIVE_STATEFUL_CPU_THREADS)
    try:
        import pyaudio
    except ImportError:
        parser.exit(1, "PyAudio is required: pip install pyaudio\n")

    if args.list_devices:
        pa = pyaudio.PyAudio()
        try:
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                if info['maxInputChannels'] > 0:
                    print(f"[{index}] {info['name']} ({int(info['defaultSampleRate'])} Hz)")
        finally:
            pa.terminate()
        return

    latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    if latest is None:
        parser.exit(1, f"No checkpoint found in {args.checkpoint_dir}\n")
    model = build_model()
    model(tf.zeros((1, 1, CHUNK)))
    status = tf.train.Checkpoint(model=model).restore(latest)
    status.expect_partial()
    status.assert_existing_objects_matched()
    if LIVE_STATEFUL_BLOCK_SECONDS <= 0:
        parser.error("LIVE_STATEFUL_BLOCK_SECONDS must be positive")
    if LIVE_STATEFUL_QUEUE_SECONDS <= 0:
        parser.error("LIVE_STATEFUL_QUEUE_SECONDS must be positive")

    pa = pyaudio.PyAudio()
    stream = None
    try:
        info = pa.get_device_info_by_index(args.device)
        rate = int(info['defaultSampleRate'])
        block_size = max(1, round(rate * LIVE_STATEFUL_BLOCK_SECONDS))
        decoder = StreamingDecoder(model, rate)
        # Trace inference before capture so initialization cannot overflow input.
        decoder.inference.forward(
            tf.zeros((1, 1, CHUNK)), tuple(decoder.inference.states)
        )
        capture = AudioCapture(
            max(1, ceil(LIVE_STATEFUL_QUEUE_SECONDS / LIVE_STATEFUL_BLOCK_SECONDS)),
            pyaudio.paContinue, pyaudio.paInputOverflow
        )
        stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=rate, input=True,
            input_device_index=args.device, frames_per_buffer=block_size,
            stream_callback=capture.callback, start=False
        )
        print(f"Loaded {latest}")
        print(f"Capturing {info['name']} at {rate} Hz; model rate {FRAMERATE} Hz")
        print(f"Capture block: {block_size / rate:.3f}s; input latency: {stream.get_input_latency():.3f}s")
        print("Stateful decoding; press Ctrl+C to stop.\nDecoded: ", end='', flush=True)
        stream.start_stream()
        slow_warning_printed = False
        while True:
            try:
                audio, discontinuity = capture.read()
            except queue.Empty:
                if not stream.is_active():
                    raise RuntimeError("Audio input stream stopped unexpectedly")
                continue
            if discontinuity:
                decoder.reset()
                print(f"\nAudio was lost: {discontinuity}; decoder state reset.", file=sys.stderr)
            started = time.perf_counter()
            text = decoder.process(audio)
            elapsed = time.perf_counter() - started
            if elapsed > len(audio) / rate and not slow_warning_printed:
                print(
                    f"\nInference took {elapsed:.3f}s for {len(audio) / rate:.3f}s of audio; "
                    "decoding may not keep up with capture.",
                    file=sys.stderr
                )
                slow_warning_printed = True
            if text:
                print(text, end='', flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except OSError:
                pass
            finally:
                try:
                    stream.close()
                except OSError:
                    pass
        pa.terminate()


if __name__ == '__main__':
    main()
