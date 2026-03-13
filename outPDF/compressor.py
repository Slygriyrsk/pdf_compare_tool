"""
MODULE 2 — compressor.py
=========================
Compression: delta + zstd.
CVR audio design decision:
  Full 20-min CVR is physically impossible over 7kbps HF (≈77 min needed).
  We send: 20-min telemetry (fits in ~2 min) + last 60-sec CVR snippet (1 min).
  Total: ~3 min << 10-min budget. CVR full recording stays on physical FDR.

Bundle binary format:
  [MAGIC 4B "A717"][version 1B][bundle_id 16B][flight_id 8B]
  [timestamp_utc 8B][event_type 1B][pre_event_sec 4B][post_event_sec 4B]
  [n_params 2B][algo 4B][orig_size 8B][comp_size 8B][sha256 32B][payload]
"""

import os, struct, time, hashlib, uuid
import numpy as np
import zstandard as zstd

MAGIC        = b"A717"
VERSION      = 1
ALGO_ZDLT    = b"ZDLT"
HEADER_FMT   = ">4sB16s8sQBIIH4sQQ32s"
HEADER_SIZE  = struct.calcsize(HEADER_FMT)
ZSTD_LEVEL   = 19
HF_BPS       = 7_000
TX_LIMIT_SEC = 600

EVT_NORMAL   = 0
EVT_DISTRESS = 1
EVT_POST     = 2

CVR_SAMPLE_RATE = 4000   # 4kHz — voice intelligible, low BW


def delta_encode(data: bytes) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return bytes(np.diff(arr, prepend=arr[0]).astype(np.uint8))


def delta_decode(data: bytes) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    return bytes((np.cumsum(arr) & 0xFF).astype(np.uint8))


def simulate_cvr_audio(duration_sec: int, sample_rate: int = CVR_SAMPLE_RATE) -> bytes:
    """
    Simulate 8-bit PCM mono CVR audio at 4kHz.
    ~60% silence (zeros), ~40% speech-like noise.
    At 4kHz 8-bit: 4000 B/sec raw; delta+zstd gives ~5-6x -> ~700 B/sec.
    60 sec fits in ~42KB compressed.
    """
    rng = np.random.default_rng(0xC0FFEE)
    n   = duration_sec * sample_rate
    s   = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        sil = int(rng.integers(sample_rate, sample_rate * 4))
        spe = int(rng.integers(sample_rate // 2, sample_rate * 3))
        pos += sil
        if pos >= n: break
        end = min(pos + spe, n)
        s[pos:end] = rng.standard_normal(end - pos).astype(np.float32) * 0.025
        pos = end
    # Quantise to 8-bit unsigned (0-255, silence=128)
    u8 = np.clip((s + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    return bytes(u8)


def compress_payload(telemetry_raw: bytes, cvr_raw: bytes = b"",
                     level: int = ZSTD_LEVEL) -> tuple:
    """
    Pack telemetry + CVR with 4B length prefixes, delta-encode, zstd compress.
    Returns (compressed_bytes, metrics_dict).
    """
    import time as _t
    t0 = _t.perf_counter()
    combined = (struct.pack(">I", len(telemetry_raw)) + telemetry_raw
                + struct.pack(">I", len(cvr_raw)) + cvr_raw)
    t1 = _t.perf_counter()
    delta = delta_encode(combined)
    t2 = _t.perf_counter()
    cctx  = zstd.ZstdCompressor(level=level, threads=-1)
    comp  = cctx.compress(delta)
    t3 = _t.perf_counter()

    orig  = len(combined)
    csz   = len(comp)
    cr    = orig / max(csz, 1)
    tx    = csz * 8 / HF_BPS
    return comp, {
        "orig_bytes": orig, "telem_bytes": len(telemetry_raw),
        "cvr_bytes": len(cvr_raw), "comp_bytes": csz, "cr": cr,
        "tx_sec": tx, "tx_min": tx / 60, "meets_10min": tx <= TX_LIMIT_SEC,
        "delta_ms": (t2-t1)*1000, "zstd_ms": (t3-t2)*1000,
        "total_ms": (t3-t0)*1000, "sha256_hex": hashlib.sha256(comp).hexdigest(),
    }


def decompress_payload(comp: bytes) -> tuple:
    dctx     = zstd.ZstdDecompressor()
    combined = delta_decode(dctx.decompress(comp))
    off = 0
    tl  = struct.unpack(">I", combined[off:off+4])[0]; off += 4
    tel = combined[off:off+tl]; off += tl
    cl  = struct.unpack(">I", combined[off:off+4])[0]; off += 4
    cvr = combined[off:off+cl]
    return tel, cvr


def pack_bundle(comp_payload: bytes, flight_id: str = "FLT0001",
                event_type: int = EVT_NORMAL, pre_event_sec: int = 1200,
                post_event_sec: int = 0, n_params: int = 88,
                timestamp: int = None) -> bytes:
    if timestamp is None: timestamp = int(time.time())
    bid  = uuid.uuid4().bytes
    fid  = flight_id.encode()[:8].ljust(8, b"\x00")
    sha  = hashlib.sha256(comp_payload).digest()
    hdr  = struct.pack(HEADER_FMT, MAGIC, VERSION, bid, fid,
                       timestamp, event_type, pre_event_sec, post_event_sec,
                       n_params, ALGO_ZDLT, 0, len(comp_payload), sha)
    return hdr + comp_payload


def unpack_bundle(data: bytes) -> dict:
    assert data[:4] == MAGIC, f"Bad magic: {data[:4]!r}"
    f = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    (magic, ver, bid, fid, ts, evt, pre, post,
     npar, algo, orig, csz, sha_stored) = f
    payload = data[HEADER_SIZE:HEADER_SIZE + csz]
    assert hashlib.sha256(payload).digest() == sha_stored, "SHA256 mismatch"
    return {
        "version": ver, "bundle_id": bid.hex(),
        "flight_id": fid.rstrip(b"\x00").decode(),
        "timestamp": ts, "event_type": evt,
        "pre_event_sec": pre, "post_event_sec": post,
        "n_params": npar, "algo": algo.rstrip(b"\x00").decode(),
        "comp_size": csz, "payload": payload,
        "sha256": sha_stored.hex(), "integrity": "OK",
    }


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from arinc_generator import generate_telemetry, BUFFER_SEC
    raw, _, _, _ = generate_telemetry(BUFFER_SEC)
    cvr = simulate_cvr_audio(60)   # 60-sec snippet
    comp, m = compress_payload(raw, cvr)
    print(f"Telem: {m['telem_bytes']:,}B  CVR(60s): {m['cvr_bytes']:,}B")
    print(f"Compressed: {m['comp_bytes']:,}B  CR:{m['cr']:.2f}x  Tx:{m['tx_min']:.2f}min")
    print(f"Within 10min budget: {m['meets_10min']}")
    # round-trip
    t,c = decompress_payload(comp)
    print(f"Round-trip: telem={'OK' if t==raw else 'FAIL'}  cvr={'OK' if c==cvr else 'FAIL'}")
