#!/usr/bin/env python3
"""
timedomain_2d_dynamic_seed.py
------------------------------
2D Time-domain voice scrambler using segment matrix row/column permutation,
with per-frame dynamic seeds derived from the master key and frame index.
"""

import argparse
import numpy as np
from scipy.io.wavfile import read, write
from scipy.signal import butter, filtfilt
import hashlib

SEGMENT_MS = 30  # Segment duration in milliseconds

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def derive_seed_from_key_and_frame(key_bytes, frame_num, purpose):
    """
    Derive a 32-bit seed from the master key, frame index, and purpose.
    Uses SHA-256, but only the first 4 bytes (to fit within 32-bit seed for np.random).
    """
    hasher = hashlib.sha256()
    hasher.update(key_bytes)
    hasher.update(frame_num.to_bytes(4, 'big'))
    hasher.update(purpose.encode())
    seed_bytes = hasher.digest()
    seed_int = int.from_bytes(seed_bytes[:4], 'big')  # Use only first 4 bytes
    return seed_int

def generate_permutation_from_seed(n, seed):
    rng = np.random.RandomState(seed)
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm

def main():
    parser = argparse.ArgumentParser(description="2D Time-Domain Voice Scrambler with Per-Frame Dynamic Seeds")
    parser.add_argument("--action", choices=["scramble", "descramble"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", required=True, help="128-bit hex key (32 hex characters)")
    parser.add_argument("--block_size", type=int, default=15, help="Segments per matrix row/block")
    args = parser.parse_args()

    key_hex = args.key.strip()
    if len(key_hex) != 32:
        raise ValueError("Key must be exactly 32 hex characters (128 bits).")
    key_bytes = bytes.fromhex(key_hex)

    sr, audio = read(args.input)
    if audio.ndim > 1:
        audio = audio[:, 0]  # Use mono channel

    segment_len = int((SEGMENT_MS / 1000.0) * sr)
    num_segments = len(audio) // segment_len
    segments = audio[:num_segments * segment_len].reshape(num_segments, segment_len)

    block_size = args.block_size

    # Pad if needed to fill last block completely
    pad_needed = (block_size - (num_segments % block_size)) % block_size
    if pad_needed > 0:
        segments = np.vstack([segments, np.zeros((pad_needed, segment_len), dtype=segments.dtype)])
        num_segments = len(segments)

    num_blocks = num_segments // block_size

    def scramble_frame(frame_segments, frame_num):
        # frame_segments shape: (block_size, segment_len)
        row_seed = derive_seed_from_key_and_frame(key_bytes, frame_num, 'row')
        col_seed = derive_seed_from_key_and_frame(key_bytes, frame_num, 'col')

        row_perm = generate_permutation_from_seed(num_blocks, row_seed)
        col_perm = generate_permutation_from_seed(block_size, col_seed)

        # Permute columns within the current block:
        permuted = frame_segments[col_perm, :]
        return permuted

    def descramble_frame(frame_segments, frame_num):
        row_seed = derive_seed_from_key_and_frame(key_bytes, frame_num, 'row')
        col_seed = derive_seed_from_key_and_frame(key_bytes, frame_num, 'col')

        col_perm = generate_permutation_from_seed(block_size, col_seed)
        inv_col = np.argsort(col_perm)
        unscrambled = frame_segments[inv_col, :]
        return unscrambled

    # Reshape into matrix of blocks: (num_blocks, block_size, segment_len)
    matrix = segments.reshape(num_blocks, block_size, segment_len)

    if args.action == "scramble":
        # Apply per-frame (block) column permutation
        scrambled_blocks = []
        for frame_idx in range(num_blocks):
            block = matrix[frame_idx]
            scrambled_block = scramble_frame(block, frame_idx)
            scrambled_blocks.append(scrambled_block)
        scrambled_matrix = np.stack(scrambled_blocks)

        # Apply row permutation on blocks with static seed derived from key
        row_seed = int.from_bytes(hashlib.sha256(key_bytes + b'row').digest()[:4], 'big')
        row_perm = generate_permutation_from_seed(num_blocks, row_seed)
        scrambled_matrix = scrambled_matrix[row_perm]

        processed_segments = scrambled_matrix.reshape(-1, segment_len)

    else:  # descramble
        # Inverse row permutation on blocks
        row_seed = int.from_bytes(hashlib.sha256(key_bytes + b'row').digest()[:4], 'big')
        row_perm = generate_permutation_from_seed(num_blocks, row_seed)
        inv_row = np.argsort(row_perm)

        descrambled_matrix = matrix[inv_row]

        descrambled_blocks = []
        for frame_idx in range(num_blocks):
            block = descrambled_matrix[frame_idx]
            descrambled_block = descramble_frame(block, frame_idx)
            descrambled_blocks.append(descrambled_block)
        processed_segments = np.stack(descrambled_blocks).reshape(-1, segment_len)

    # Apply bandpass filter
    filtered = butter_bandpass_filter(processed_segments.flatten(), 100, 3500, sr)
    write(args.output, sr, filtered.astype(np.int16))
    print(f"{args.action.capitalize()} complete. Output written to {args.output}")

if __name__ == "__main__":
    main()