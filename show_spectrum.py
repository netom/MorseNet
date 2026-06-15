#!/usr/bin/env python3
"""
Display the spectrogram of a test WAV file.

Usage:
    ./show_spectrum.py [wavfile]
"""

import sys
import numpy as np
import scipy.io.wavfile
import scipy.fft
import matplotlib.pyplot as plt
from pathlib import Path

from config import FRAMERATE, CHUNK

def fft_spectrogram(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """
    Compute a standard STFT-based spectrogram.

    Returns an array of shape (frame_size//2 + 1, num_frames) with magnitudes.
    """
    num_frames = (len(audio) - frame_size) // hop_size + 1
    window = np.hanning(frame_size)

    frames = np.stack([
        audio[i * hop_size: i * hop_size + frame_size] * window
        for i in range(num_frames)
    ])  # (num_frames, frame_size)

    fft_frames = np.fft.rfft(frames, axis=1)  # (num_frames, frame_size//2 + 1)
    return np.abs(fft_frames).T               # (freq_bins, num_frames)

def main():
    wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test0.wav")

    if not wav_path.exists():
        print(f"Error: '{wav_path}' not found.")
        sys.exit(1)

    rate, data = scipy.io.wavfile.read(wav_path)
    data = data.astype(np.float32)

    if rate != FRAMERATE:
        print(f"Warning: sample rate {rate} Hz differs from expected {FRAMERATE} Hz")

    duration = len(data) / rate
    print(f"File   : {wav_path}  ({duration:.2f} s, {rate} Hz)")

    frame_size = CHUNK        # 256 samples = 32 ms at 8 kHz  (matches network chunk size)
    hop_size   = CHUNK // 2   # 50 % overlap

    spec_fft = fft_spectrogram(data, frame_size, hop_size)  # (frame_size//2+1, num_frames)

    # Frequency axes
    fft_freq = np.fft.rfftfreq(frame_size, d=1.0 / rate)                 # Hz
    time_axis = np.arange(spec_fft.shape[1]) * hop_size / rate           # seconds

    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [1, 1]})

    log_fft = 20 * np.log10(spec_fft + 1e-6)

    # --- FFT spectrogram ---
    ax = axes[0]
    img2 = ax.pcolormesh(time_axis, fft_freq, log_fft,
                         shading="auto", cmap="inferno",
                         vmin=np.percentile(log_fft, 5),
                         vmax=np.percentile(log_fft, 99.5))
    fig.colorbar(img2, ax=ax, label="dB")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("FFT spectrogram (log magnitude)")

    # --- Time-averaged spectrum ---
    ax = axes[1]
    ax.plot(fft_freq, 20 * np.log10(np.mean(spec_fft, axis=1) + 1e-6),
            linewidth=0.8, label="FFT", alpha=0.8)
    ax.axvspan(400, 800, alpha=0.15, color="orange", label="signal band")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_title("Time-averaged spectra")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
