#!/usr/bin/env python3
"""
a3_8band_inv_v4.py – 8-band scrambler with pre-filtering and spectral separation.
Includes DC offset removal, high-pass filtering, and phase-preserving spectral inversion.
"""

import argparse
import hashlib
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, medfilt

def make_bands(fs):
    edges = np.linspace(250, 3400, 9)  # 8 bands
    return [(edges[i], edges[i+1]) for i in range(8)]

def design_bp(low, high, fs, order=6):  # Reduced order to minimize ringing
    ny = 0.5 * fs
    sos = butter(order, [low / ny, high / ny], btype='band', output='sos')
    return sos

def prng_bits(key_hex: str, window_idx: int):
    h = hashlib.sha256()
    h.update(bytes.fromhex(key_hex))
    h.update(window_idx.to_bytes(4, 'big'))
    digest = h.digest()
    seed = int.from_bytes(digest[:4], 'big')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(8)
    inv_bytes = digest[4:12]
    if len(inv_bytes) < 8:
        inv_bytes = inv_bytes + b'\x00' * (8 - len(inv_bytes))
    inv_bits_array = np.unpackbits(np.frombuffer(inv_bytes, dtype=np.uint8))[:8]
    inv_bits = inv_bits_array.astype(bool)
    return perm, inv_bits

def invert_spectrum_phase_preserving(x):
    X = np.fft.fft(x)
    magnitude = np.abs(X)
    phase = np.angle(X)
    # Invert magnitude spectrum
    inv_magnitude = magnitude[::-1]
    # Reconstruct spectrum with original phase
    inv_spectrum = inv_magnitude * np.exp(1j * phase)
    inv_x = np.fft.ifft(inv_spectrum).real
    return inv_x

def translate_band(x, fs, src_band, dst_band):
    src_lo, src_hi = src_band
    dst_lo, dst_hi = dst_band
    shift = ((dst_lo + dst_hi) - (src_lo + src_hi)) / 2.0
    t = np.arange(len(x)) / fs
    mixed = x * np.cos(2 * np.pi * shift * t)
    sos = design_bp(dst_lo, dst_hi, fs)
    return sosfilt(sos, mixed)

def process_window(chunk, fs, key_hex, idx, action, bands):
    slices = []
    for low, high in bands:
        sos = design_bp(low, high, fs)
        slices.append(sosfilt(sos, chunk))
    perm, inv_bits = prng_bits(key_hex, idx)
    if action == 'descramble':
        perm = np.argsort(perm)
    # Enforce minimum 75% inverted bands
    invert_flags = inv_bits.copy()
    min_invert = int(0.75 * len(bands))
    num_invert = np.sum(invert_flags)
    if num_invert < min_invert:
        indices_to_invert = np.where(~invert_flags)[0]
        np.random.shuffle(indices_to_invert)
        for i in indices_to_invert[:min_invert - num_invert]:
            invert_flags[i] = True
    out = np.zeros_like(chunk)
    for src_idx, dst_idx in enumerate(perm):
        sig = slices[src_idx]
        if invert_flags[src_idx]:
            sig = invert_spectrum_phase_preserving(sig)
        out += translate_band(sig, fs, bands[src_idx], bands[dst_idx])
    # Apply median filter to reduce ringing
    out = medfilt(out, kernel_size=3)
    return out

def preprocess_signal(signal, fs):
    # Remove DC offset
    signal = signal - np.mean(signal)
    # High-pass filter at 20 Hz
    sos_hp = butter(4, 20 / (0.5 * fs), btype='highpass', output='sos')
    signal = sosfilt(sos_hp, signal)
    return signal

def main():
    parser = argparse.ArgumentParser(description='8-band scrambler with pre-filtering and spectral separation.')
    parser.add_argument('--action', choices=['scramble', 'descramble'], required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--key', required=True, help='12 hex characters (6 bytes)')
    parser.add_argument('--rate', type=int, choices=range(1, 31), default=4,
                        help='bands swaps per second (1–30)')
    args = parser.parse_args()

    if len(args.key) != 12:
        raise ValueError('Key must be exactly 12 hex characters (6 bytes).')

    # Read input audio
    sig, fs = sf.read(args.input)
    if sig.ndim > 1:
        sig = sig[:, 0]

    # Preprocess input: remove DC and high-pass filter
    sig = preprocess_signal(sig, fs)

    win_len = fs // args.rate
    total = int(np.ceil(len(sig) / win_len))
    padded = np.pad(sig, (0, total * win_len - len(sig)), mode='constant')

    bands = make_bands(fs)
    out = np.zeros_like(padded)

    for idx in range(total):
        chunk = padded[idx * win_len : (idx + 1) * win_len]
        processed_chunk = process_window(chunk, fs, args.key, idx, args.action, bands)
        out[idx * win_len : (idx + 1) * win_len] = processed_chunk

    # Write output
    sf.write(args.output, out[:len(sig)], fs)
    print(f'{args.action.capitalize()} complete → {args.output}')

if __name__ == '__main__':
    main()