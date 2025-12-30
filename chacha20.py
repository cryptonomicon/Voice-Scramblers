import argparse
import wave
import os
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def chacha20_encrypt_decrypt(input_file, output_file, key_hex, nonce_hex, action):
    key = bytes.fromhex(key_hex)

    if len(key) != 32:
        raise ValueError("Key must be 256 bits (64 hex characters)")

    with open(input_file, 'rb') as f:
        wav_data = f.read()

    header = wav_data[:44]  # standard WAV header size
    pcm_data = wav_data[44:]

    if action == 'encrypt':
        if nonce_hex is None:
            nonce = secrets.token_bytes(16)
            print(f"Generated nonce: {nonce.hex().upper()}")
        else:
            nonce = bytes.fromhex(nonce_hex)
            if len(nonce) != 16:
                raise ValueError("Nonce must be 128 bits (32 hex characters)")

        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        encryptor = cipher.encryptor()
        processed_data = encryptor.update(pcm_data)

        # Prepend nonce as 16-byte prefix to PCM data
        with open(output_file, 'wb') as f:
            f.write(header + nonce + processed_data)

    elif action == 'decrypt':
        if len(pcm_data) < 16:
            raise ValueError("Encrypted PCM data too short to contain nonce.")

        embedded_nonce = pcm_data[:16]
        encrypted_data = pcm_data[16:]

        algorithm = algorithms.ChaCha20(key, embedded_nonce)
        cipher = Cipher(algorithm, mode=None, backend=default_backend())
        decryptor = cipher.decryptor()
        recovered_data = decryptor.update(encrypted_data)

        with open(output_file, 'wb') as f:
            f.write(header + recovered_data)

    print(f"{action.capitalize()}ed WAV written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encrypt/Decrypt WAV PCM data using ChaCha20")
    parser.add_argument('--action', choices=['encrypt', 'decrypt'], required=True, help='Action to perform')
    parser.add_argument('--input', required=True, help='Input WAV file')
    parser.add_argument('--output', required=True, help='Output WAV file')
    parser.add_argument('--key', required=True, help='256-bit hex key (64 hex digits)')
    parser.add_argument('--nonce', help='128-bit hex nonce (32 hex digits). Optional for encryption, ignored for decryption')

    args = parser.parse_args()
    chacha20_encrypt_decrypt(args.input, args.output, args.key, args.nonce, args.action)
