"""
cvr_audio.py
============
Drop-in replacement for simulate_cvr_audio() in compressor.py.
Load a real audio file (m4a, mp3, wav, aac, ogg, flac …)
and convert it to the same 4 kHz / 8-bit / mono PCM bytes
the compressor expects.

Requirements:
    pip install pydub
    ffmpeg must be on PATH  (brew install ffmpeg / apt install ffmpeg)

Usage — 3 ways:

  1. Direct import in compressor.py:
       from cvr_audio import load_cvr_audio
       cvr_raw = load_cvr_audio("cockpit_recording.m4a")

  2. Clip to the last N seconds (pre-event window):
       cvr_raw = load_cvr_audio("cockpit_recording.m4a", last_n_sec=60)

  3. CLI test:
       python cvr_audio.py cockpit_recording.m4a
"""

import os
import sys
import struct
import numpy as np
import zstandard

try:
    from pydub import AudioSegment
except ImportError:
    raise ImportError(
        "pydub is required for real CVR audio.\n"
        "Install with:  pip install pydub\n"
        "Also install ffmpeg: https://ffmpeg.org/download.html"
    )

# ── Constants (must match compressor.py) ─────────────────────────────────────
CVR_SAMPLE_RATE = 4_000      # 4 kHz — voice intelligible, minimal bandwidth
CVR_SAMPLE_WIDTH = 1         # 8-bit (1 byte per sample)
CVR_CHANNELS     = 1         # mono
HF_BPS           = 7_000     # 7 kbps link budget


def load_cvr_audio(filepath: str,
                   last_n_sec: int = None,
                   target_rate: int = CVR_SAMPLE_RATE) -> bytes:
    """
    Load any audio file supported by ffmpeg and return
    raw 8-bit unsigned PCM bytes at 4 kHz mono.

    Args:
        filepath    : path to the audio file (.m4a, .mp3, .wav, .aac, …)
        last_n_sec  : if given, keep only the LAST N seconds of the file
                      (e.g. last_n_sec=60 gives the 60-sec pre-crash window)
        target_rate : output sample rate in Hz (default 4000)

    Returns:
        bytes — raw 8-bit unsigned PCM, ready to pass to compress_payload()

    What this does under the hood:
        m4a/aac (or any format)
          → ffmpeg decode to PCM
          → convert to mono  (mix down all channels)
          → resample to 4 kHz  (voice intelligible; halves size vs 8 kHz)
          → quantise to 8-bit unsigned  (halves size vs 16-bit)
        Total size: 4000 bytes/sec raw → ~750 bytes/sec after delta+zstd
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CVR audio file not found: {filepath!r}")

    print(f"[CVR] Loading: {filepath}")
    audio = AudioSegment.from_file(filepath)   # ffmpeg handles m4a / aac / mp3 etc.

    print(f"[CVR] Original : {audio.duration_seconds:.1f} s  "
          f"{audio.frame_rate} Hz  "
          f"{audio.channels} ch  "
          f"{audio.sample_width * 8}-bit  "
          f"({len(audio.raw_data):,} bytes raw)")

    # ── Optional: keep only the last N seconds ────────────────────────────────
    if last_n_sec is not None and audio.duration_seconds > last_n_sec:
        trim_ms = last_n_sec * 1000
        audio   = audio[-trim_ms:]    # pydub slice syntax: audio[-60000:] = last 60 sec
        print(f"[CVR] Trimmed  : keeping last {last_n_sec}s")

    # ── Convert: mono → 4 kHz → 8-bit ────────────────────────────────────────
    audio = audio.set_channels(CVR_CHANNELS)        # mono mix-down
    audio = audio.set_frame_rate(target_rate)       # resample to 4 kHz
    audio = audio.set_sample_width(CVR_SAMPLE_WIDTH)  # 8-bit quantisation

    raw = audio.raw_data

    duration_sec = len(audio) / 1000.0
    print(f"[CVR] Converted: {duration_sec:.1f} s  "
          f"{target_rate} Hz mono 8-bit  "
          f"{len(raw):,} bytes")
    print(f"[CVR] Est. compressed size : ~{len(raw)//5:,} bytes  "
          f"(~{len(raw)*8/5/HF_BPS/60:.2f} min Tx @ 7 kbps)")

    return raw


def get_cvr_info(filepath: str) -> dict:
    """
    Return metadata about the audio file without loading it fully.
    Useful for checking duration before deciding how much to trim.
    """
    audio = AudioSegment.from_file(filepath)
    return {
        "filepath":       filepath,
        "duration_sec":   round(audio.duration_seconds, 2),
        "sample_rate_hz": audio.frame_rate,
        "channels":       audio.channels,
        "bit_depth":      audio.sample_width * 8,
        "raw_bytes":      len(audio.raw_data),
        "size_mb":        round(len(audio.raw_data) / 1e6, 2),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cvr_audio.py <audio_file.m4a> [last_n_sec]")
        print("Example: python cvr_audio.py cockpit.m4a 60")
        sys.exit(1)

    path       = sys.argv[1]
    n_sec      = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    # Show file info
    info = get_cvr_info(path)
    print(f"\n[INFO] {info}")

    # Load and convert
    raw = load_cvr_audio(path, last_n_sec=n_sec)

    # Test compression (delta + zstd — same as compressor.py)
    arr   = np.frombuffer(raw, dtype=np.uint8)
    delta = np.diff(arr, prepend=arr[0]).astype(np.uint8).tobytes()
    cctx  = zstandard.ZstdCompressor(level=19)
    comp  = cctx.compress(delta)

    print(f"\n[COMPRESS] Raw:        {len(raw):,} bytes")
    print(f"[COMPRESS] Compressed: {len(comp):,} bytes")
    print(f"[COMPRESS] Ratio:      {len(raw)/len(comp):.2f}x")
    print(f"[COMPRESS] Tx @ 7kbps: {len(comp)*8/HF_BPS/60:.2f} min")
    print(f"\n[OK] cvr_audio.py pipeline verified for: {path}")
