import argparse
import numpy as np
import hashlib
from scipy.fft import fft, ifft
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import librosa
import soundfile as sf

# --- Filtering Utilities ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def remove_dc_offset(data):
    mean_value = np.mean(data)
    return data - mean_value

def normalize_audio(data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val
    return data

# --- Spectral Permutation Utilities ---

def generate_seed(key, block_idx):
    """
    Generate a seed from key and block index (not per-frame index),
    so that one permutation is reused for a block of frames.
    """
    combined = key + block_idx.to_bytes(4, 'big')
    seed_bytes = hashlib.sha256(combined).digest()
    seed = int.from_bytes(seed_bytes[:8], 'big')
    return seed

def generate_permutation(n_bins, seed):
    rng = np.random.default_rng(seed)
    perm = np.arange(n_bins)
    rng.shuffle(perm)
    return perm

# --- Core Processing ---

def process_frame(frame, key, frame_idx, n_bins, scramble=True, frames_per_perm=16):
    """
    Process a single frame with a spectral permutation.

    frames_per_perm controls how many frames share the same permutation.
    A value of 16 means one permutation for 16 consecutive frames.
    """
    window = np.hanning(len(frame))
    windowed = frame * window
    spectrum = fft(windowed)

    # Use block index (frame_idx // frames_per_perm) instead of raw frame_idx
    block_idx = frame_idx // frames_per_perm
    seed = generate_seed(key, block_idx)
    perm = generate_permutation(n_bins, seed)

    bin_size = len(spectrum) // n_bins
    bins = [spectrum[i * bin_size:(i + 1) * bin_size] for i in range(n_bins)]

    if scramble:
        # Apply permutation: output bin i comes from original bin perm[i]
        permuted_bins = [bins[p] for p in perm]
    else:
        # Invert permutation to recover original bin order
        inv_perm = np.argsort(perm)
        permuted_bins = [bins[inv_perm[p]] for p in range(n_bins)]

    fft_permuted = np.concatenate(permuted_bins)
    time_frame = np.real(ifft(fft_permuted))
    return time_frame

def process_audio(data, key, frame_size, hop_size, n_bins, scramble=True, frames_per_perm=16):
    data = data.astype(np.float32)
    length = len(data)

    # Calculate number of frames
    n_frames = int(np.ceil((length - frame_size) / hop_size)) + 1
    padded_length = (n_frames - 1) * hop_size + frame_size
    padded_data = np.zeros(padded_length, dtype=np.float32)
    padded_data[:length] = data

    output = np.zeros(padded_length, dtype=np.float32)
    window_sum = np.zeros(padded_length, dtype=np.float32)
    window = np.hanning(frame_size)

    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        end = start + frame_size
        frame = padded_data[start:end]
        processed = process_frame(
            frame,
            key,
            frame_idx,
            n_bins,
            scramble=scramble,
            frames_per_perm=frames_per_perm
        )
        output[start:end] += processed * window
        window_sum[start:end] += window

    # Normalize overlap-add
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    output = output[:length]

    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 32767

    return np.clip(output, -32768, 32767).astype(np.int16), output

# --- Post-processing: filter, DC removal, normalization ---

def post_process_audio(float_audio, sr):
    # Remove DC offset
    audio_no_dc = remove_dc_offset(float_audio)
    # Bandpass filter (250-3400 Hz)
    filtered = bandpass_filter(audio_no_dc, 250, 3400, sr)
    # Normalize to prevent clipping
    normalized = normalize_audio(filtered)
    # Scale to int16 range
    scaled = normalized * 32767
    return np.clip(scaled, -32768, 32767).astype(np.int16)

# --- Main script ---

def main():
    parser = argparse.ArgumentParser(
        description="Spectral Scrambler with filtering, DC removal, normalization"
    )
    parser.add_argument('--action', choices=['scramble', 'descramble'], required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--bins',
        type=int,
        choices=[64, 128, 256, 512, 1024],
        required=True
    )
    parser.add_argument(
        '--key',
        type=str,
        required=True,
        help='Hex key up to 48 hex characters'
    )
    parser.add_argument('--frame_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=128)
    parser.add_argument(
        "--delay_stretch",
        action='store_true',
        help="Apply long slow speech effect"
    )
    parser.add_argument(
        "--stretch_factor",
        type=float,
        default=1.4,
        help="Time stretch factor (>1.0 for slower)"
    )
    parser.add_argument(
        "--delay_seconds",
        type=float,
        default=0.5,
        help="Delay in seconds"
    )
    parser.add_argument(
        "--frames_per_perm",
        type=int,
        default=16,
        help="Number of frames that share the same permutation"
    )
    args = parser.parse_args()

    # Read input WAV
    sr, data = wavfile.read(args.input)
    if data.ndim > 1:
        data = np.mean(data, axis=1).astype(np.int16)

    # Prepare key bytes
    key_str = args.key.strip()
    if key_str.startswith('0x'):
        key_str = key_str[2:]
    key_str = key_str.zfill(48)
    try:
        key_bytes = bytes.fromhex(key_str)
    except ValueError:
        key_bytes = key_str.encode()

    scramble = (args.action == 'scramble')

    # Process audio (permutation held constant for frames_per_perm frames)
    processed_int16, processed_float = process_audio(
        data,
        key_bytes,
        args.frame_size,
        args.hop_size,
        args.bins,
        scramble=scramble,
        frames_per_perm=args.frames_per_perm
    )

    # For descramble, apply post-processing
    if not scramble:
        print("Applying post-processing: bandpass filter, DC removal, normalization...")
        final_int16 = post_process_audio(processed_float, sr)
    else:
        final_int16 = processed_int16

    # Save the main output
    wavfile.write(args.output, sr, final_int16)
    print(f"{args.action.capitalize()} complete! Saved to {args.output}")

    # Optional: Long slow speech effect
    if args.delay_stretch:
        print("Applying slow stretch and delay...")
        audio_data, sr_loaded = librosa.load(args.output, sr=None)
        print(f"Loaded audio duration: {len(audio_data) / sr_loaded:.2f}s")
        stretched = librosa.effects.time_stretch(
            audio_data,
            rate=1.0 / args.stretch_factor
        )
        delay_samples = int(sr_loaded * args.delay_seconds)
        delayed = np.pad(stretched, (delay_samples, 0), mode='constant')
        final_path = args.output.replace('.wav', '_long.wav')
        sf.write(final_path, delayed, sr_loaded)
        print(f"Long, stretched speech saved to: {final_path}")

if __name__ == '__main__':
    main()
