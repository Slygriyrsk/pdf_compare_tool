"""
MODULE 5 — pipeline.py
=======================
Top-level orchestrator. Ties everything together.

System flow:
  ┌─────────────────────────────────────────────────────────────┐
  │ AIRPLANE SIDE                                               │
  │                                                             │
  │  arinc_generator  →  circular_buffer  →  distress_trigger  │
  │         ↓                                                   │
  │  compressor (delta+zstd)                                    │
  │         ↓                                                   │
  │  packet_framer (Ed25519 sign + chunk)                       │
  │         ↓                                                   │
  │  quic_transport (send over QUIC, track metrics)             │
  └─────────────────────────────────────────────────────────────┘
                           ↕ 7 kbps QUIC
  ┌─────────────────────────────────────────────────────────────┐
  │ GROUND STATION SIDE                                         │
  │                                                             │
  │  quic_transport (receive, verify, save)                     │
  │         ↓                                                   │
  │  compressor (decompress)                                    │
  │         ↓                                                   │
  │  decoded telemetry + CVR audio files                        │
  └─────────────────────────────────────────────────────────────┘

Distress modes:
  NORMAL   — heartbeat: send a 1-frame summary every 60s (minimal BW)
  DISTRESS — triggered event: send 20-min pre-buffer + ongoing real-time
  POST     — after distress cleared: finalize and send post-event data

DTN concept applied:
  - Pre-event buffer is always maintained in RAM (circular buffer)
  - On distress trigger: snapshot the buffer, compress, chunk, sign, send
  - If link drops mid-send: receiver remembers which chunks it has
    (via manifest's chunk list); sender only re-sends missing ones
  - Post-distress: continue sending until buffer_post_sec elapsed

Run as:
  python pipeline.py airplane  --peer <multiaddr>  [--distress-at 30]
  python pipeline.py ground    --port 8000

  --distress-at N : simulate distress event N seconds after start
                    (for testing without real trigger hardware)
"""

import os
import sys
import json
import time
import struct
import argparse
import hashlib
import threading
from collections import deque
from typing import Optional

import numpy as np
import trio

# ── Local module imports ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from arinc_generator  import generate_telemetry, BUFFER_SEC, PARAM_DEFS, HF_BPS
from compressor       import (compress_payload, decompress_payload, pack_bundle,
                               unpack_bundle, simulate_cvr_audio,
                               EVT_NORMAL, EVT_DISTRESS, EVT_POST)
from packet_framer    import build_chunks, Signer, Verifier, DEFAULT_CHUNK
from quic_transport   import (send_bundle, run_receiver, make_host,
                               quic_addrs, TransportMetrics)

PRE_BUFFER_SEC  = 1200   # 20 min pre-event buffer
POST_BUFFER_SEC = 300    # 5 min post-event data
OUTPUT_DIR      = "./pipeline_output"


# ── Circular telemetry buffer ──────────────────────────────────────────────────
class CircularBuffer:
    """
    Rolling window of the last N seconds of raw telemetry bytes.
    Thread-safe. Airplane side continuously appends; on distress we snapshot.
    """

    def __init__(self, max_seconds: int = PRE_BUFFER_SEC):
        self.max_sec = max_seconds
        self._lock   = threading.Lock()
        # each entry: (timestamp, frame_raw_bytes)
        self._buf: deque = deque()
        self._total_bytes = 0

    def append(self, frame_bytes: bytes, ts: float = None):
        ts = ts or time.time()
        with self._lock:
            self._buf.append((ts, frame_bytes))
            self._total_bytes += len(frame_bytes)
            # evict entries older than max_sec
            cutoff = ts - self.max_sec
            while self._buf and self._buf[0][0] < cutoff:
                _, old = self._buf.popleft()
                self._total_bytes -= len(old)

    def snapshot(self) -> bytes:
        """Return concatenation of all buffered frame bytes (most recent window)."""
        with self._lock:
            return b"".join(fb for _, fb in self._buf)

    def duration_sec(self) -> float:
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            return self._buf[-1][0] - self._buf[0][0]

    def __len__(self):
        return len(self._buf)


# ── Distress trigger ───────────────────────────────────────────────────────────
class DistressTrigger:
    """
    Monitors for distress conditions and fires a callback.
    In real system: reads ACARS, FDR flags, pilot input.
    For prototype: simulates a trigger at a configurable time.
    """

    def __init__(self, callback, simulate_at_sec: Optional[float] = None):
        self.callback        = callback
        self.simulate_at_sec = simulate_at_sec
        self._fired          = False
        self._start          = time.time()

    def check(self, param_data: dict = None) -> bool:
        """
        Call periodically. Returns True if distress just triggered.
        Checks:
          1. Simulated trigger (for testing)
          2. Extreme pitch/roll (would be from real params in production)
        """
        if self._fired:
            return False

        now = time.time()

        # Simulated trigger
        if self.simulate_at_sec and (now - self._start) >= self.simulate_at_sec:
            reason = f"SIMULATED distress at t+{self.simulate_at_sec:.0f}s"
            self._fire(reason)
            return True

        # Real parameter check (prototype: just check pitch if data available)
        if param_data:
            pitch = param_data.get("Pitch_deg", None)
            if pitch is not None and abs(pitch[-1]) > 30:
                self._fire(f"PITCH EXCEEDANCE: {pitch[-1]:.1f}°")
                return True

        return False

    def _fire(self, reason: str):
        self._fired = True
        print(f"\n[DISTRESS] *** TRIGGER: {reason} ***")
        self.callback(reason)


# ── Airplane side ──────────────────────────────────────────────────────────────
def run_airplane(peer_addr: str,
                 port: int = 8001,
                 distress_at_sec: float = None,
                 flight_id: str = "FLT0001"):
    """
    Airplane-side pipeline:
      1. Generate 20-min telemetry (pre-event buffer) + CVR
      2. Wait for distress trigger
      3. On trigger: compress + chunk + sign + send
      4. Continue sending post-event data
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'█'*65}")
    print(f"  AIRPLANE SIDE — Flight {flight_id}")
    print(f"  Pre-buffer: {PRE_BUFFER_SEC}s  Post-buffer: {POST_BUFFER_SEC}s")
    print(f"  Peer: {peer_addr}")
    print(f"{'█'*65}")

    # ── Generate pre-event telemetry (full 20-min buffer) ────────────────────
    print(f"\n[PLANE] Generating {PRE_BUFFER_SEC}s pre-event telemetry...")
    t0 = time.time()
    raw_telem, frame_arr, slot_map, param_data = generate_telemetry(PRE_BUFFER_SEC)
    print(f"[PLANE] Telemetry ready in {time.time()-t0:.1f}s  "
          f"({len(raw_telem):,} bytes)")

    # ── Generate CVR audio ────────────────────────────────────────────────────
    print(f"[PLANE] Generating {PRE_BUFFER_SEC}s CVR audio...")
    cvr_raw = simulate_cvr_audio(PRE_BUFFER_SEC)
    print(f"[PLANE] CVR raw: {len(cvr_raw):,} bytes")

    # ── Print brief frame summary ─────────────────────────────────────────────
    _print_frame_summary(frame_arr, slot_map, max_frames=5)

    # ── Compress ──────────────────────────────────────────────────────────────
    print(f"\n[PLANE] Compressing (delta + zstd)...")
    comp, metrics = compress_payload(raw_telem, cvr_raw)
    _print_compression_report(metrics)

    # ── Build signed bundle ───────────────────────────────────────────────────
    signer  = Signer()
    bundle  = pack_bundle(
        comp,
        flight_id=flight_id,
        event_type=EVT_DISTRESS,
        pre_event_sec=PRE_BUFFER_SEC,
        post_event_sec=0,
        n_params=len(PARAM_DEFS),
    )
    print(f"[PLANE] Bundle size: {len(bundle):,} bytes")

    # Save bundle locally for reference
    bundle_path = f"{OUTPUT_DIR}/{flight_id}_bundle.a717"
    with open(bundle_path, "wb") as f:
        f.write(bundle)
    print(f"[PLANE] Bundle saved locally: {bundle_path}")

    # Save public key so ground can verify
    pk_path = f"{OUTPUT_DIR}/{flight_id}_pubkey.hex"
    with open(pk_path, "w") as f:
        f.write(signer.public_key_bytes.hex())
    print(f"[PLANE] Public key saved: {pk_path}")

    # ── Chunk and sign ────────────────────────────────────────────────────────
    manifest_chunk, data_chunks = build_chunks(
        bundle, signer, chunk_size=DEFAULT_CHUNK, flight_id=flight_id
    )

    # ── Simulate distress trigger then send ───────────────────────────────────
    if distress_at_sec:
        elapsed = time.time() - t0
        wait    = max(0, distress_at_sec - elapsed)
        print(f"\n[PLANE] Waiting {wait:.0f}s to simulate distress trigger...")
        time.sleep(wait)
        print(f"[PLANE] *** DISTRESS TRIGGERED ***")

    # ── Send over QUIC ────────────────────────────────────────────────────────
    tx_metrics = trio.run(
        send_bundle,
        manifest_chunk,
        data_chunks,
        peer_addr,
        port,
        flight_id,
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    report       = tx_metrics.report()
    metrics_path = f"{OUTPUT_DIR}/{flight_id}_tx_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[PLANE] Metrics saved: {metrics_path}")
    _print_budget_summary(metrics, report)


# ── Ground side ────────────────────────────────────────────────────────────────
def run_ground(port: int = 8000, pubkey_hex: str = None, flight_id: str = "FLT0001"):
    """
    Ground station side:
      1. Start QUIC receiver
      2. Receive + verify chunks
      3. Reassemble and decompress
      4. Decode telemetry back to engineering values
      5. Save everything to disk
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'█'*65}")
    print(f"  GROUND STATION — Listening on port {port}")
    print(f"{'█'*65}")

    # Load public key if provided
    if pubkey_hex:
        verifier = Verifier(bytes.fromhex(pubkey_hex))
        print(f"[GROUND] Using provided public key for signature verification")
    elif os.path.exists(f"{OUTPUT_DIR}/{flight_id}_pubkey.hex"):
        with open(f"{OUTPUT_DIR}/{flight_id}_pubkey.hex") as f:
            verifier = Verifier(bytes.fromhex(f.read().strip()))
        print(f"[GROUND] Loaded public key from {OUTPUT_DIR}/{flight_id}_pubkey.hex")
    else:
        # No key: create a dummy verifier that always passes
        # In production, this should reject
        print(f"[GROUND] WARNING: No public key — signature verification disabled")
        verifier = _NullVerifier()

    trio.run(run_receiver, port, verifier, 1)

    # After receive: find the saved bundle and decode it
    import glob
    bundles = sorted(glob.glob(f"./received_bundles/*.a717"))
    if not bundles:
        print("[GROUND] No bundles received.")
        return

    latest = bundles[-1]
    print(f"\n[GROUND] Decoding bundle: {latest}")
    with open(latest, "rb") as f:
        bundle_data = f.read()

    info = unpack_bundle(bundle_data)
    print(f"[GROUND] Bundle: flight={info['flight_id']}  "
          f"event={info['event_type']}  chunks_orig={info['comp_size']:,}B")

    telem_raw, cvr_raw = decompress_payload(info["payload"])
    print(f"[GROUND] Decompressed: telem={len(telem_raw):,}B  cvr={len(cvr_raw):,}B")

    # Save raw telemetry
    telem_path = f"{OUTPUT_DIR}/{info['flight_id']}_telemetry.bin"
    with open(telem_path, "wb") as f:
        f.write(telem_raw)

    cvr_path = f"{OUTPUT_DIR}/{info['flight_id']}_cvr.pcm"
    with open(cvr_path, "wb") as f:
        f.write(cvr_raw)

    print(f"[GROUND] Telemetry saved: {telem_path}")
    print(f"[GROUND] CVR audio saved: {cvr_path}")
    print(f"[GROUND] Done. SHA256 integrity: {info['integrity']}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _print_frame_summary(frame_arr, slot_map, max_frames: int = 5):
    total_sfs = frame_arr.shape[0]
    n_frames  = total_sfs // 4
    print(f"\n[FRAMES] Total: {n_frames} frames  ({total_sfs} subframes)")
    print(f"  {'Frame':>5}  {'SF':>3}  {'Sync':>6}  "
          + "  ".join(f"{a.param_name[:10]:>10}" for a in slot_map[:5]))
    print("  " + "─" * 80)
    for fr in range(min(max_frames, n_frames)):
        for sf in range(4):
            sf_idx = fr * 4 + sf
            row    = frame_arr[sf_idx]
            vals   = []
            for a in slot_map[:5]:
                for (sfx, slot) in a.slots:
                    if sfx == sf and slot < 256:
                        vals.append(row[slot])
                        break
                else:
                    vals.append(0)
            lbl = f"F{fr+1:03d}" if sf == 0 else "    "
            sync_oct = oct(row[0])
            print(f"  {lbl:>5} SF{sf+1} {sync_oct:>6}  "
                  + "  ".join(f"{v:>10d}" for v in vals))
    if n_frames > max_frames:
        print(f"  ... {n_frames - max_frames} more frames ...")


def _print_compression_report(m: dict):
    print(f"\n  ┌── Compression Report ─────────────────────────────┐")
    print(f"  │  Telemetry raw  : {m['telem_bytes']:>12,} bytes              │")
    print(f"  │  CVR raw        : {m['cvr_bytes']:>12,} bytes              │")
    print(f"  │  Combined raw   : {m['orig_bytes']:>12,} bytes              │")
    print(f"  │  Compressed     : {m['comp_bytes']:>12,} bytes              │")
    print(f"  │  Ratio          : {m['cr']:>12.2f}×                      │")
    print(f"  │  Tx @ 7 kbps    : {m['tx_min']:>12.2f} min                 │")
    fits = '✓ FITS in 10-min window' if m['meets_10min'] else '✗ EXCEEDS window'
    print(f"  │  Budget         : {fits:<30}  │")
    print(f"  │  Encode time    : {m['total_ms']:>12.0f} ms                 │")
    print(f"  └──────────────────────────────────────────────────────┘")


def _print_budget_summary(comp_metrics: dict, tx_metrics: dict):
    print(f"\n  ┌── Mission Budget Summary ─────────────────────────┐")
    print(f"  │  BW limit       : 7,000 bps                        │")
    print(f"  │  Time limit     : 10 min = 600 sec                 │")
    print(f"  │  Compressed     : {comp_metrics['comp_bytes']:>12,} bytes              │")
    print(f"  │  Est. Tx time   : {comp_metrics['tx_min']:>12.2f} min                 │")
    print(f"  │  Actual Tx time : {tx_metrics['elapsed_sec']:>12.2f} sec                 │")
    print(f"  │  Throughput     : {tx_metrics['throughput_bps']:>12.0f} bps                 │")
    loss = tx_metrics['loss_pct']
    print(f"  │  Loss rate      : {loss:>12.2f}%                       │")
    print(f"  └──────────────────────────────────────────────────────┘")


class _NullVerifier:
    """Allows bypass of sig verification when no key is available (test only)."""
    def verify(self, data: bytes, sig: bytes) -> bool:
        return True


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ARINC 717 Flight Data Transmission System"
    )
    parser.add_argument("mode", choices=["airplane", "ground", "test"],
                        help="Run as airplane sender, ground receiver, or self-test")
    parser.add_argument("--peer",        default=None,
                        help="Peer multiaddr (airplane mode): "
                             "/ip4/x.x.x.x/udp/8000/quic-v1/p2p/<peer_id>")
    parser.add_argument("--port",        type=int, default=8000,
                        help="Listen port (ground mode, default 8000)")
    parser.add_argument("--sender-port", type=int, default=8001,
                        help="Sender local port (airplane mode, default 8001)")
    parser.add_argument("--distress-at", type=float, default=10,
                        help="Simulate distress trigger after N seconds (default 10)")
    parser.add_argument("--flight-id",   default="FLT0001",
                        help="Flight identifier string")
    parser.add_argument("--pubkey",      default=None,
                        help="Hex public key for signature verification (ground mode)")
    args = parser.parse_args()

    if args.mode == "airplane":
        if not args.peer:
            parser.error("--peer is required for airplane mode")
        run_airplane(
            peer_addr=args.peer,
            port=args.sender_port,
            distress_at_sec=args.distress_at,
            flight_id=args.flight_id,
        )

    elif args.mode == "ground":
        run_ground(
            port=args.port,
            pubkey_hex=args.pubkey,
            flight_id=args.flight_id,
        )

    elif args.mode == "test":
        _self_test()


def _self_test():
    """
    End-to-end self test: generates data, compresses, chunks, signs,
    reassembles, decompresses — without actual network.
    """
    print(f"\n{'═'*65}")
    print(f"  SELF TEST — Full pipeline (no network)")
    print(f"{'═'*65}")

    # 1. Generate (2 min for speed)
    duration = 120
    print(f"\n[TEST] Generating {duration}s telemetry (88 params)...")
    raw, frame_arr, slot_map, param_data = generate_telemetry(duration)
    cvr = simulate_cvr_audio(duration)

    _print_frame_summary(frame_arr, slot_map, max_frames=3)

    # 2. Compress
    print(f"\n[TEST] Compressing...")
    comp, metrics = compress_payload(raw, cvr)
    _print_compression_report(metrics)

    # 3. Bundle
    bundle = pack_bundle(comp, flight_id="TST0001",
                         event_type=EVT_DISTRESS, pre_event_sec=duration)

    # 4. Chunk + sign
    signer   = Signer()
    verifier = Verifier(signer.public_key_bytes)
    mf_chunk, data_chunks = build_chunks(bundle, signer)

    # 5. Simulate receive (all chunks, no loss)
    from packet_framer import parse_chunk, reassemble
    all_parsed = [parse_chunk(mf_chunk)] + [parse_chunk(c) for c in data_chunks]
    data_only  = [c for c in all_parsed if not c["is_manifest"]]
    result, report = reassemble(data_only, verifier)

    print(f"\n[TEST] Reassemble: {report}")

    # 6. Decompress and verify
    info       = unpack_bundle(result)
    telem_back, cvr_back = decompress_payload(info["payload"])

    telem_ok = telem_back == raw
    cvr_ok   = cvr_back   == cvr
    print(f"\n[TEST] Telemetry round-trip: {'PASS ✓' if telem_ok else 'FAIL ✗'}")
    print(f"[TEST] CVR round-trip:       {'PASS ✓' if cvr_ok else 'FAIL ✗'}")

    # 7. Compute total wire overhead
    total_wire = sum(len(c) for c in data_chunks) + len(mf_chunk)
    overhead   = (total_wire - len(bundle)) / len(bundle) * 100
    tx_sec     = total_wire * 8 / HF_BPS

    print(f"\n[TEST] Wire stats:")
    print(f"  Bundle size    : {len(bundle):,} bytes")
    print(f"  Total wire     : {total_wire:,} bytes  (overhead {overhead:.1f}%)")
    print(f"  Tx @ 7 kbps    : {tx_sec/60:.2f} min")
    print(f"  Chunk count    : {len(data_chunks)}")
    print(f"  Sig overhead   : 64B per chunk = {64*len(data_chunks):,}B total")

    print(f"\n{'═'*65}")
    print(f"  SELF TEST COMPLETE")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
