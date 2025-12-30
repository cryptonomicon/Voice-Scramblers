import argparse
import numpy as np
import soundfile as sf
from scipy.fft import dct, idct
from scipy.signal import butter, filtfilt
import hmac, hashlib

"""
Galaxy – Secure DCT‑based audio scrambler (HMAC-based PRNG)
• Key length: up to 128 bits (32 hex digits)
• Uses HMAC‑SHA256(key, frame_index) to generate a secure per‑frame seed
• Reversible without storing permutations
"""

# --------------------------------------------------------------------------- #
# Key handling
# --------------------------------------------------------------------------- #

def normalize_key(hex_key: str) -> bytes:
    """Normalize hex key to 128-bit (16-byte) value."""
    if len(hex_key) > 32:
        raise ValueError("Key may be at most 32 hex digits (128 bits).")
    if len(hex_key) < 8:
        print("⚠️  Warning: Key is short and may be weak (<32 bits entropy).")
    return bytes.fromhex(hex_key.zfill(32))


# --------------------------------------------------------------------------- #
# Secure permutation generator using HMAC-SHA256
# --------------------------------------------------------------------------- #

def generate_secure_permutation(n: int, key_bytes: bytes, frame_index: int) -> np.ndarray:
    """Generate secure pseudo-random permutation for a given frame index."""
    msg = frame_index.to_bytes(4, 'big')
    digest = hmac.new(key_bytes, msg, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:4], 'big')
    rng = np.random.default_rng(seed)
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm

# --------------------------------------------------------------------------- #
# Frame helper filters
# --------------------------------------------------------------------------- #

def remove_dc(frame: np.ndarray) -> np.ndarray:
    """Subtract mean to eliminate DC bias."""
    return frame - frame.mean()

def lowpass(signal: np.ndarray, sr: int, cutoff: float = 4000.0, order: int = 6) -> np.ndarray:
    """Zero-phase Butterworth LPF."""
    nyq = 0.5 * sr
    b, a = butter(order, cutoff / nyq, btype='low')
    pad_len = max(3 * (len(b) - 1), 1)
    return filtfilt(b, a, signal, padlen=pad_len)

# --------------------------------------------------------------------------- #
# Core processor (scramble / descramble)
# --------------------------------------------------------------------------- #

def process_audio(input_file: str, output_file: str, action: str, segment_ms: int, hex_key: str, skip_lpf: bool) -> None:
    data, sr = sf.read(input_file)
    if data.ndim > 1:  # Down-mix stereo to mono
        data = data.mean(axis=1)

    seg_len = int(segment_ms * sr / 1000)
    if seg_len < 2:
        raise ValueError("segment_ms too small for the given sample rate")

    step = seg_len // 2  # 50% overlap
    window = np.hanning(seg_len)

    pad = (step - (len(data) - seg_len) % step) % step
    data = np.pad(data, (0, pad), mode='constant')

    out = np.zeros_like(data)
    overlap_weight = np.zeros_like(data)

    key_bytes = normalize_key(hex_key)

    for idx, start in enumerate(range(0, len(data) - seg_len + 1, step)):
        frame = data[start:start + seg_len] * window
        frame = remove_dc(frame)

        coeffs = dct(frame, norm='ortho')
        perm = generate_secure_permutation(len(coeffs), key_bytes, idx)

        if action == 'scramble':
            coeffs = coeffs[perm]
        else:  # descramble
            coeffs_descr = np.empty_like(coeffs)
            coeffs_descr[perm] = coeffs
            coeffs = coeffs_descr

        recon = idct(coeffs, norm='ortho')
        out[start:start + seg_len] += recon * window
        overlap_weight[start:start + seg_len] += window ** 2

    out[overlap_weight > 0] /= overlap_weight[overlap_weight > 0]

    if action == 'descramble' and not skip_lpf:
        out = lowpass(out, sr, cutoff=4000.0)

    sf.write(output_file, out, sr)
    print(f"{action.capitalize()}d audio saved to {output_file}")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Galaxy Secure DCT audio scrambler (HMAC-based)")
    p.add_argument("--action", choices=["scramble", "descramble"], required=True, help="scramble | descramble")
    p.add_argument("--input", required=True, help="Input WAV file")
    p.add_argument("--output", required=True, help="Output WAV file")
    p.add_argument("--segment_ms", type=int, default=40, help="Segment length in ms (default 40)")
    p.add_argument("--key", required=True, help="Hexadecimal key (8–32 hex digits = up to 128 bits)")
    p.add_argument("--no_post_lpf", action="store_true", help="Skip 4 kHz low-pass after descramble")
    args = p.parse_args()

    process_audio(args.input, args.output, args.action, args.segment_ms, args.key, args.no_post_lpf)
