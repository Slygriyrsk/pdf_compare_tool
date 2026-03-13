"""
cvr_loader.py
=============
Load audio into raw PCM bytes — NO ffmpeg required.

Supports:
    .wav  → scipy.io.wavfile  (standard library path, no external deps)
    .bin  → raw PCM binary    (you specify sample_rate / channels / bit_depth)
    .pcm  → same as .bin

The output is always: 16-bit signed PCM, mono, at whatever sample rate
the source has (we don't resample — just convert format and mix to mono).
This is then compressed with delta + zstd before sending.

Usage examples:
    # WAV file
    from cvr_loader import load_audio_file
    raw_bytes, meta = load_audio_file("cockpit.wav")

    # Raw .bin file (must tell us the format)
    raw_bytes, meta = load_audio_file(
        "audio.bin",
        sample_rate=16000,
        channels=1,
        bit_depth=16
    )

    # Our own pipeline's CVR output (4kHz 8-bit mono raw)
    raw_bytes, meta = load_audio_file(
        "FLT0001_cvr.pcm",
        sample_rate=4000,
        channels=1,
        bit_depth=8
    )

CLI test:
    python cvr_loader.py cockpit.wav
    python cvr_loader.py audio.bin --rate 16000 --channels 1 --bits 16
"""

import os
import sys
import struct
import wave
import numpy as np
import zstandard


# ── WAV loader (no external deps) ─────────────────────────────────────────────
def _load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file using stdlib wave module.
    Returns (samples_int16, sample_rate).
    Handles 8-bit, 16-bit, 24-bit, 32-bit WAV.
    Converts to mono by averaging channels.
    """
    from scipy.io import wavfile   # scipy always available
    rate, data = wavfile.read(path)

    # Convert to float32 for processing
    if data.dtype == np.uint8:           # 8-bit WAV is unsigned
        data = (data.astype(np.float32) - 128) / 128.0
    elif data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        data = data.astype(np.float32)

    # Mix stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Back to int16 for transmission
    samples = np.clip(data * 32767, -32767, 32767).astype(np.int16)
    return samples, rate


# ── Raw PCM / .bin loader ──────────────────────────────────────────────────────
def _load_raw_pcm(path: str,
                  sample_rate: int,
                  channels: int,
                  bit_depth: int) -> tuple[np.ndarray, int]:
    """
    Load raw binary PCM file (no header).
    User must specify format parameters.
    """
    raw = open(path, "rb").read()

    if bit_depth == 8:
        # 8-bit PCM is unsigned (0-255), convert to signed
        arr  = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (arr - 128) / 128.0
    elif bit_depth == 16:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif bit_depth == 32:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 8, 16, or 32.")

    # De-interleave channels and mix to mono
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    samples = np.clip(data * 32767, -32767, 32767).astype(np.int16)
    return samples, sample_rate


# ── Public API ────────────────────────────────────────────────────────────────
def load_audio_file(path: str,
                    sample_rate: int = 16000,
                    channels: int = 1,
                    bit_depth: int = 16,
                    last_n_sec: int = None) -> tuple[bytes, dict]:
    """
    Load audio from file. Returns (raw_bytes, metadata_dict).

    raw_bytes : 16-bit signed PCM mono, network byte order (big-endian)
                This is what gets compressed and sent over QUIC.

    metadata_dict : sample_rate, n_samples, duration_sec, bit_depth,
                    original_path — these ride along in the bundle header
                    so the receiver can reconstruct the audio properly.

    Args:
        path        : file path (.wav, .bin, .pcm)
        sample_rate : only used for .bin/.pcm (WAV reads it from header)
        channels    : only used for .bin/.pcm
        bit_depth   : only used for .bin/.pcm (8, 16, or 32)
        last_n_sec  : if set, keep only the last N seconds of audio
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path!r}")

    ext = os.path.splitext(path)[1].lower()
    print(f"[AUDIO] Loading: {path}  ({os.path.getsize(path):,} bytes on disk)")

    if ext == ".wav":
        samples, rate = _load_wav(path)
    elif ext in (".bin", ".pcm", ""):
        samples, rate = _load_raw_pcm(path, sample_rate, channels, bit_depth)
    else:
        # Try WAV as fallback for unknown extensions
        try:
            samples, rate = _load_wav(path)
        except Exception:
            # Fall back to raw PCM interpretation
            print(f"[AUDIO] Unknown extension {ext!r} — treating as raw {bit_depth}-bit PCM @ {sample_rate}Hz")
            samples, rate = _load_raw_pcm(path, sample_rate, channels, bit_depth)

    n_samples    = len(samples)
    duration_sec = n_samples / rate

    print(f"[AUDIO] Loaded : {duration_sec:.2f} s  {rate} Hz mono 16-bit  "
          f"{n_samples:,} samples  {n_samples*2:,} bytes")

    # Trim to last N seconds if requested
    if last_n_sec is not None and duration_sec > last_n_sec:
        keep     = int(last_n_sec * rate)
        samples  = samples[-keep:]
        n_samples    = len(samples)
        duration_sec = n_samples / rate
        print(f"[AUDIO] Trimmed to last {last_n_sec}s: {n_samples:,} samples  "
              f"{n_samples*2:,} bytes")

    # Serialize as big-endian int16 (consistent across platforms)
    raw_bytes = samples.astype(">i2").tobytes()

    meta = {
        "sample_rate":   rate,
        "n_samples":     n_samples,
        "duration_sec":  round(duration_sec, 3),
        "bit_depth":     16,
        "channels":      1,
        "encoding":      "PCM_S16BE",   # signed 16-bit big-endian
        "source_file":   os.path.basename(path),
    }

    # Estimate compression
    arr   = np.frombuffer(raw_bytes, dtype=np.uint8)
    delta = np.diff(arr, prepend=arr[0]).astype(np.uint8).tobytes()
    est   = zstandard.ZstdCompressor(level=3).compress(delta[:min(len(delta), 65536)])
    est_ratio = min(len(delta), 65536) / max(len(est), 1)
    est_total = len(raw_bytes) / est_ratio
    print(f"[AUDIO] Est. compressed size: ~{int(est_total):,} bytes  "
          f"(~{est_ratio:.1f}× ratio)")

    return raw_bytes, meta


def audio_to_wav(raw_bytes: bytes, sample_rate: int, out_path: str):
    """
    Write received audio bytes back to a standard WAV file.
    raw_bytes must be signed 16-bit big-endian PCM (what load_audio_file produces).
    """
    samples = np.frombuffer(raw_bytes, dtype=">i2").astype(np.int16)
    from scipy.io import wavfile
    wavfile.write(out_path, sample_rate, samples)
    print(f"[AUDIO] Saved WAV: {out_path}  ({os.path.getsize(out_path):,} bytes)")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Test audio file loading (no ffmpeg)")
    p.add_argument("file",                      help="Audio file (.wav / .bin / .pcm)")
    p.add_argument("--rate",     type=int, default=16000, help="Sample rate for .bin files (default 16000)")
    p.add_argument("--channels", type=int, default=1,     help="Channels for .bin files (default 1)")
    p.add_argument("--bits",     type=int, default=16,    help="Bit depth for .bin files (default 16)")
    p.add_argument("--last",     type=int, default=None,  help="Keep only last N seconds")
    p.add_argument("--save-wav", default=None,            help="Save converted audio as WAV to this path")
    args = p.parse_args()

    raw, meta = load_audio_file(
        args.file,
        sample_rate=args.rate,
        channels=args.channels,
        bit_depth=args.bits,
        last_n_sec=args.last,
    )

    print(f"\n[META] {meta}")
    print(f"[BYTES] Raw bytes to transmit: {len(raw):,}")

    # Full compression test
    import zstandard
    arr   = np.frombuffer(raw, dtype=np.uint8)
    delta = np.diff(arr, prepend=arr[0]).astype(np.uint8).tobytes()
    comp  = zstandard.ZstdCompressor(level=19).compress(delta)
    print(f"\n[COMPRESS] Raw: {len(raw):,} B  →  Compressed: {len(comp):,} B  "
          f"CR: {len(raw)/len(comp):.2f}×")

    if args.save_wav:
        audio_to_wav(raw, meta["sample_rate"], args.save_wav)
