#!/usr/bin/env python3

"""
Live audio decoding script for morse code using TensorFlow 2.x.

This script captures live audio from a microphone and continuously decodes
morse code in real-time using a trained model.
"""

from __future__ import generator_stop

import tensorflow as tf
import numpy as np
import sys
from pathlib import Path
from collections import deque
import time

# Suppress TensorFlow warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import *
from model import create_cw_model, ctc_decode, decoded_to_text

try:
    import pyaudio
except ImportError:
    print("Error: pyaudio not installed")
    print("Install with: pip install pyaudio")
    sys.exit(1)


class LiveDecoder:
    """Live audio decoder for morse code."""

    def __init__(self, checkpoint_dir='./model_use', buffer_seconds=12):
        """
        Initialize live decoder.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            buffer_seconds: Length of audio buffer to decode (seconds)
        """
        self.buffer_seconds = buffer_seconds
        self.buffer_chunks = (FRAMERATE * buffer_seconds) // CHUNK
        self.audio_buffer = deque(maxlen=self.buffer_chunks)
        self.last_decoded = ""

        print("="*70)
        print("Live Morse Code Decoder - TensorFlow 2.x")
        print("="*70)
        print(f"Buffer length: {buffer_seconds} seconds")
        print(f"Sample rate: {FRAMERATE} Hz")
        print(f"Chunk size: {CHUNK} samples")
        print("="*70)

        # Load model
        print("\nLoading model...")
        self.model = self._load_model(checkpoint_dir)

        if self.model is None:
            raise ValueError(f"Failed to load model from {checkpoint_dir}")

        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()

        # List available audio devices
        print(f"\nAvailable audio devices:")
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")

        # Open audio stream
        try:
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=FRAMERATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE,
                stream_callback=self._audio_callback
            )
            print(f"\nUsing device: {DEVICE}")
        except Exception as e:
            print(f"\nError opening audio device: {e}")
            print(f"Try changing DEVICE in config.py")
            raise

    def _load_model(self, checkpoint_dir):
        """Load model from checkpoint."""
        model = create_cw_model(
            max_timesteps=None,  # Variable length
            num_features=CHUNK,
            recurrent_layer_depth=3,  # STANDARDIZED to 3 layers
            recurrent_layer_width=128,
            num_classes=NUM_CLASSES
        )

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
            return None

        checkpoint = tf.train.Checkpoint(model=model)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if latest_checkpoint:
            status = checkpoint.restore(latest_checkpoint)
            status.expect_partial()
            print(f"Model loaded from: {latest_checkpoint}")
        else:
            print(f"Error: No checkpoint found in {checkpoint_dir}")
            return None

        return model

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        self.audio_buffer.append(audio_chunk)

        return (in_data, pyaudio.paContinue)

    def decode_buffer(self):
        """
        Decode current audio buffer.

        Returns:
            Decoded text string, or None if buffer not full
        """
        if len(self.audio_buffer) < self.buffer_chunks:
            return None

        # Prepare audio
        audio = np.array(list(self.audio_buffer), dtype=np.float32)

        # Normalize
        audio = (audio - np.mean(audio)) / np.std(audio)

        # Add batch dimension [1, timesteps, features]
        audio = np.expand_dims(audio, axis=0)

        # Forward pass
        logits = self.model(audio, training=False)

        # Decode with greedy decoder (faster for real-time)
        decoded, _ = ctc_decode(
            logits,
            sequence_length=[len(self.audio_buffer)],
            use_beam_search=False  # Use greedy for speed
        )

        # Convert to text
        decoded_text = decoded_to_text(decoded[0], MORSE_CHR)

        return decoded_text[0]

    def start(self):
        """Start live decoding."""
        print("\n" + "="*70)
        print("Starting live decoding...")
        print("Press Ctrl+C to stop")
        print("="*70)
        print("\nWaiting for audio buffer to fill...")

        self.stream.start_stream()

        try:
            buffer_filled = False

            while True:
                time.sleep(0.5)  # Decode every 500ms

                # Check if buffer is full
                if not buffer_filled:
                    if len(self.audio_buffer) >= self.buffer_chunks:
                        buffer_filled = True
                        print("Buffer filled. Starting decoding...\n")
                    else:
                        progress = len(self.audio_buffer) / self.buffer_chunks * 100
                        print(f"\rBuffer: {progress:.0f}%", end='', flush=True)
                    continue

                # Decode
                decoded = self.decode_buffer()

                if decoded is not None:
                    # Only print if changed
                    if decoded != self.last_decoded:
                        print(f"\rDecoded: {decoded}", end='', flush=True)
                        self.last_decoded = decoded

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("Stopping...")
            print("="*70)

        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'stream') and self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        if hasattr(self, 'pa') and self.pa:
            try:
                self.pa.terminate()
            except:
                pass


def main(args):
    """Main function."""

    if len(args) > 1 and args[1] in ['-h', '--help']:
        print("Usage: ./tensorflow_lstm_ctc_live.py")
        print("\nLive morse code decoder using microphone input")
        print("\nConfiguration:")
        print(f"  Device: Edit DEVICE in config.py (currently: {DEVICE})")
        print(f"  Sample rate: {FRAMERATE} Hz")
        print(f"  Buffer: 12 seconds")
        sys.exit(0)

    try:
        decoder = LiveDecoder(checkpoint_dir='./model_use')
        decoder.start()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
