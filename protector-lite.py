import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import argparse
import re

INVERSION_FREQUENCIES = [
    2105, 2218, 2341, 2431,
    2531, 2632, 2728, 2868,
    3023, 3107, 3196, 3333,
    3495, 3610, 3729, 3850,
    4000, 4096
]

MIN_FREQ = 300
MAX_FREQ = 3000


def create_bandpass_filter(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.iirfilter(
        N=order,
        Wn=[low, high],
        btype='band',
        ftype='bessel'
    )
    return b, a


def main():
    parser = argparse.ArgumentParser(description='Voice inversion with dynamic hopping.')
    parser.add_argument('--input', required=True, help='Input WAV file')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--seed', type=str, default='00000000',
                        help='Hex seed (up to 8 hex characters)')
    parser.add_argument('--key', type=int, choices=range(1, 19),
                        help='Static inversion key (1-18)')
    parser.add_argument('--hop', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Hopping mode')

    args = parser.parse_args()

    # --- Seed ---
    seed_str = args.seed.strip()
    if not re.fullmatch(r'[0-9A-Fa-f]{1,8}', seed_str):
        raise ValueError("Seed must be a hex string up to 8 characters")
    seed_int = int(seed_str.zfill(8), 16)
    np.random.seed(seed_int)

    # --- Read WAV ---
    fs, data = wav.read(args.input)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    duration_sec = len(data) / fs

    durations_ms = []
    freqs = []

    # --- Hop selection ---
    if args.hop == 0:
        if args.key is None:
            raise ValueError("Static mode requires --key")
        freq = INVERSION_FREQUENCIES[args.key - 1]
        durations_ms = [int(duration_sec * 1000)]
        freqs = [freq]
    else:
        if args.hop == 1:
            hop_min_ms, hop_max_ms = 833, 1667
        elif args.hop == 2:
            hop_min_ms, hop_max_ms = 417, 833
        else:
            hop_min_ms, hop_max_ms = 208, 417

        total_ms = 0
        while total_ms < duration_sec * 1000:
            dur = np.random.randint(hop_min_ms, hop_max_ms + 1)
            freq = INVERSION_FREQUENCIES[
                np.random.randint(0, len(INVERSION_FREQUENCIES))
            ]
            durations_ms.append(dur)
            freqs.append(freq)
            total_ms += dur

    samples_per_hop = [int(fs * ms / 1000) for ms in durations_ms]

    # --- Output ---
    processed = np.zeros_like(data)

    # Continuous phase accumulator (CRITICAL FIX)
    phase = 0.0
    start_idx = 0

    for i, hop_len in enumerate(samples_per_hop):
        end_idx = min(start_idx + hop_len, len(data))
        segment = data[start_idx:end_idx]
        if len(segment) == 0:
            break

        freq = freqs[i]
        print(f"[DEBUG] Segment {i}: {freq} Hz")

        # --- Pre-filter ---
        pre_cutoff = min(max(freq * 0.9, 80), fs / 2 - 100)
        b_lp, a_lp = signal.iirfilter(
            4,
            pre_cutoff / (0.5 * fs),
            btype='low',
            ftype='bessel'
        )
        filtered = signal.filtfilt(b_lp, a_lp, segment)

        # --- Inversion oscillator (continuous phase) ---
        t = np.arange(len(filtered)) / fs
        osc = np.cos(2 * np.pi * freq * t + phase)

        # Update phase for next hop
        phase += 2 * np.pi * freq * len(filtered) / fs
        phase = np.mod(phase, 2 * np.pi)

        inverted = -filtered * osc

        # --- Bandpass after inversion ---
        bandwidth = max(freq * 2, 300)
        lowcut = max(freq - bandwidth / 2, MIN_FREQ)
        highcut = min(freq + bandwidth / 2, MAX_FREQ)

        b_bp, a_bp = create_bandpass_filter(lowcut, highcut, fs)
        result = signal.filtfilt(b_bp, a_bp, inverted)

        processed[start_idx:end_idx] = result
        start_idx = end_idx

        if start_idx >= len(data):
            break

    # --- Normalize ---
    peak = np.max(np.abs(processed))
    if peak > 0:
        processed /= peak

    wav.write(args.output, fs, (processed * 32767).astype(np.int16))
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
