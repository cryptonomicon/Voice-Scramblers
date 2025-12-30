#!/usr/bin/env python3
"""
Unified Voice Scrambler: Variable Split-Band + 2D Time-Domain Shuffle
"""

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import hashlib
import hmac
import argparse

# Constants
SEGMENT_MS = 30

# Split-band keys (Hz)
SPLIT_BAND_KEYS = {
    1:  (350, 650, 3731),  2:  (388, 688, 3787),  3:  (428, 728, 3816),
    4:  (471, 772, 3846),  5:  (512, 813, 3906),  6:  (552, 853, 3937),
    7:  (591, 891, 3968),  8:  (636, 936, 4032),  9:  (688, 988, 4065),
    10: (736, 1037, 4132), 11: (792, 1094, 4166), 12: (857, 1157, 4273),
    13: (913, 1213, 4310), 14: (976, 1278, 4347), 15: (1050, 1351, 4424),
    16: (1135, 1436, 4504), 17: (1200, 1501, 4587), 18: (1272, 1572, 4629),
    19: (1354, 1655, 4716), 20: (1448, 1748, 4807),
    21: (1555, 1858, 4950), 22: (1680, 1984, 5050), 23: (1750, 2049, 5102),
    24: (1826, 2127, 5208), 25: (1909, 2212, 5263), 26: (2000, 2304, 5376),
    27: (2100, 2403, 5494), 28: (2210, 2512, 5555), 29: (2333, 2631, 5681),
    30: (2470, 2777, 5813), 31: (2625, 2923, 6024), 32: (2800, 3105, 6172),
}

def bandpass_filter(data, lowcut, highcut, fs):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    if high <= low:
        return data
    sos = signal.butter(6, [low, high], btype='band', output='sos')
    return signal.sosfiltfilt(sos, data)

def split_bands(data, fs, split_freq):
    transition = 200
    low_edge = max(split_freq - transition/2, 251)
    high_edge = split_freq + transition/2
    low_band  = bandpass_filter(data, 250, low_edge, fs)
    high_band = bandpass_filter(data, high_edge, 5000, fs)
    return low_band, high_band

def invert_band_hilbert(band_data, fs, fc):
    analytic = signal.hilbert(band_data)
    t = np.arange(len(band_data)) / fs
    carrier = np.exp(-1j * 2 * np.pi * fc * t)
    shifted = analytic * carrier
    return np.real(shifted)

def derive_seed_hmac(key_bytes, frame_num, purpose: str) -> int:
    message = frame_num.to_bytes(4, 'big') + purpose.encode()
    hmac_obj = hmac.new(key_bytes, message, hashlib.sha256)
    digest = hmac_obj.digest()
    return int.from_bytes(digest[:4], 'big')

def derive_vsb_hop_rate(key_bytes, frame_num, min_rate=4.0, max_rate=20.0):
    seed_bytes = hmac.new(key_bytes, b"hoprate"+frame_num.to_bytes(4, 'big'), hashlib.sha256).digest()
    val = int.from_bytes(seed_bytes[:4], 'big') / 2**32
    return min_rate + val * (max_rate - min_rate)

def derive_static_row_seed(key_bytes):
    """
    Derive a static 32-bit seed for the global row (block) permutation
    using HMAC-SHA-256.
    """
    hmac_obj = hmac.new(key_bytes, b"row_static", hashlib.sha256)
    digest = hmac_obj.digest()
    return int.from_bytes(digest[:4], 'big')

def generate_permutation_from_seed(n, seed):
    rng = np.random.RandomState(seed)
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm

def vsb_process(audio, fs, key_bytes, frame_ms=100):
    """
    Apply VSB hopping with dynamic hop rate.
    """
    audio = audio.astype(np.float32)
    frame_len = int(fs * frame_ms / 1000)
    total_frames = len(audio) // frame_len
    if total_frames == 0:
        return audio
    out = np.zeros_like(audio)

    for frame_idx in range(total_frames):
        start = frame_idx * frame_len
        end = start + frame_len
        chunk = audio[start:end]
        # derive seed for this frame
        seed = derive_seed_hmac(key_bytes, frame_idx, "vsb")
        prng = np.random.default_rng(seed)
        # derive hop rate
        hop_rate = derive_vsb_hop_rate(key_bytes, frame_idx, min_rate=4.0, max_rate=20.0)
        frames_per_hop = max(1, int(1000 / hop_rate / frame_ms))
        # pick key index
        key_index = prng.integers(1, 33)
        split_freq, f_lower, f_upper = SPLIT_BAND_KEYS[key_index]
        # process frame
        low, high = split_bands(chunk, fs, split_freq)
        inv_low  = invert_band_hilbert(low, fs, f_lower)
        inv_high = invert_band_hilbert(high, fs, f_upper)
        processed = inv_low + inv_high
        out[start:end] = processed[:len(chunk)]
    # normalize
    max_amp = np.max(np.abs(out))
    if max_amp > 0:
        out /= max_amp
    return out

def time2d_process(audio, fs, key_bytes, block_size=15, action="scramble"):
    """
    2D time permutation with static row permutation, per-frame column permutation.
    """
    audio = audio.astype(np.float32)
    orig_len = len(audio)
    segment_len = int((SEGMENT_MS / 1000) * fs)
    num_segments = len(audio) // segment_len
    if num_segments == 0:
        return audio
    segments = audio[:num_segments*segment_len].reshape(num_segments, segment_len)

    # Pad if needed
    pad_needed = (block_size - (num_segments % block_size)) % block_size
    if pad_needed > 0:
        pad = np.zeros((pad_needed, segment_len), dtype=segments.dtype)
        segments = np.vstack([segments, pad])
        num_segments = len(segments)

    num_blocks = num_segments // block_size

    def get_col_perm(frame_num):
        seed = derive_seed_hmac(key_bytes, frame_num, "col")
        return generate_permutation_from_seed(block_size, seed)

    def get_row_perm():
        seed = derive_static_row_seed(key_bytes)
        return generate_permutation_from_seed(num_blocks, seed)

    row_perm = get_row_perm()
    inv_row_perm = np.argsort(row_perm)
    col_perm_cache = {}

    def scramble_frame(frame_idx):
        seed = derive_seed_hmac(key_bytes, frame_idx, "col")
        col_perm = generate_permutation_from_seed(block_size, seed)
        col_perm_cache[frame_idx] = col_perm
        return col_perm

    def descramble_frame(frame_idx):
        col_perm = col_perm_cache.get(frame_idx)
        if col_perm is None:
            seed = derive_seed_hmac(key_bytes, frame_idx, "col")
            col_perm = generate_permutation_from_seed(block_size, seed)
            col_perm_cache[frame_idx] = col_perm
        inv_col_perm = np.argsort(col_perm)
        return inv_col_perm

    matrix = segments.reshape(num_blocks, block_size, segment_len)

    if action == "scramble":
        # scramble columns per frame
        for frame_idx in range(num_blocks):
            col_perm = scramble_frame(frame_idx)
            matrix[frame_idx] = matrix[frame_idx][col_perm]
        # permute blocks (rows)
        matrix = matrix[row_perm]
    else:
        # inverse row permutation
        matrix = matrix[inv_row_perm]
        # inverse column permutation
        for frame_idx in range(num_blocks):
            inv_col_perm = descramble_frame(frame_idx)
            matrix[frame_idx] = matrix[frame_idx][inv_col_perm]

    processed_segments = matrix.reshape(-1, segment_len)
    # trim to original length
    processed_segments = processed_segments[:(orig_len // segment_len), :]
    flattened = processed_segments.flatten()
    # optional bandpass
    filtered = bandpass_filter(flattened, 100, 3500, fs)
    filtered = filtered[:orig_len]
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Unified Voice Scrambler")
    parser.add_argument("--action", choices=["scramble", "descramble"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", required=True, help="128-bit hex key (32 hex chars)")
    parser.add_argument("--block_size", type=int, default=15)
    parser.add_argument("--frame_ms", type=int, default=100)
    args = parser.parse_args()

    key_hex = args.key.strip()
    if len(key_hex) != 32:
        raise ValueError("Key must be 32 hex characters (128 bits)")
    key_bytes = bytes.fromhex(key_hex)

    sr, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = data[:,0]
    data = data.astype(np.float32) / 32768.0
    orig_len = len(data)

    if args.action == "scramble":
        # 1. VSB with dynamic hop
        vsb_out = vsb_process(data, sr, key_bytes, frame_ms=args.frame_ms)
        # 2. Time shuffle
        final = time2d_process(vsb_out, sr, key_bytes, block_size=args.block_size, action="scramble")
    else:
        # 1. Undo time shuffle
        unshuffled = time2d_process(data, sr, key_bytes, block_size=args.block_size, action="descramble")
        # 2. Undo VSB
        final = vsb_process(unshuffled, sr, key_bytes, frame_ms=args.frame_ms)

    # final bandpass
    final = bandpass_filter(final, 100, 3500, sr)
    final = final[:orig_len]
    # clip
    final = np.clip(final, -1, 1)
    wavfile.write(args.output, sr, (final * 32767).astype(np.int16))
    print(f"{args.action} complete, output saved to {args.output}")

if __name__ == "__main__":
    main()