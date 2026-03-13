"""
sender.py
=========
Standalone sender: load telemetry .bin + audio .bin, compress,
chunk, sign with persistent key, send over QUIC, print full metrics.

No bandwidth limits. No time limits.
Focus: did every packet arrive? What was the latency? Full metric per chunk.

Run:
    python sender.py \\
        --telem  pipeline_output/FLT0001_telemetry.bin \\
        --audio  pipeline_output/FLT0001_cvr.pcm \\
        --peer   /ip4/<ground_ip>/udp/8000/quic-v1/p2p/<peer_id> \\
        --flight FLT0001

    # For raw .bin audio (need to specify format):
    python sender.py \\
        --telem  telemetry.bin \\
        --audio  cockpit_audio.bin \\
        --audio-rate 16000 --audio-bits 16 --audio-channels 1 \\
        --peer   /ip4/192.168.1.5/udp/8000/quic-v1/p2p/16Uiu2... \\
        --flight FLT0001

    # For WAV audio (format auto-detected):
    python sender.py \\
        --telem  telemetry.bin \\
        --audio  cockpit.wav \\
        --peer   /ip4/192.168.1.5/udp/8000/quic-v1/p2p/16Uiu2... \\
        --flight FLT0001
"""

import os
import sys
import json
import time
import struct
import hashlib
import secrets
import argparse
import numpy as np
import zstandard
import trio

from multiaddr import Multiaddr
from libp2p import new_host
from libp2p.crypto.secp256k1 import create_new_key_pair
from libp2p.peer.peerinfo import info_from_p2p_addr
from libp2p.security.noise.transport import (
    PROTOCOL_ID as NOISE_PROTOCOL_ID,
    Transport as NoiseTransport,
)
from libp2p.utils.address_validation import get_available_interfaces
from libp2p.rcmgr import ResourceLimits, new_resource_manager
import nacl.signing
import nacl.encoding

sys.path.insert(0, os.path.dirname(__file__))
from cvr_loader  import load_audio_file
from keygen      import load_private_key, PRIVATE_KEY_FILE

# ── Protocol IDs ──────────────────────────────────────────────────────────────
PROTO_DATA = "/arinc717/data/1.0.0"
PROTO_ACK  = "/arinc717/ack/1.0.0"

CHUNK_SIZE     = 60_000   # 60 KB per chunk — no BW limit, use large chunks
CONNECT_RETRY  = 3
ACK_TIMEOUT    = 60       # seconds per chunk before retransmit

OUTPUT_DIR = "./pipeline_output"


# ── Wire framing (4-byte length prefix) ───────────────────────────────────────
async def send_msg(stream, data: bytes):
    await stream.write(struct.pack(">I", len(data)) + data)

async def recv_msg(stream) -> bytes:
    hdr = b""
    while len(hdr) < 4:
        c = await stream.read(4 - len(hdr))
        if not c: raise EOFError("stream closed before header")
        hdr += c
    n = struct.unpack(">I", hdr)[0]
    data = b""
    while len(data) < n:
        c = await stream.read(n - len(data))
        if not c: raise EOFError(f"truncated: {len(data)}/{n}")
        data += c
    return data


# ── Compression ────────────────────────────────────────────────────────────────
def delta_encode(data: bytes) -> bytes:
    arr = np.frombuffer(data, dtype=np.uint8)
    return bytes(np.diff(arr, prepend=arr[0]).astype(np.uint8))

def compress(data: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=19, threads=-1).compress(delta_encode(data))


# ── Chunk + sign ───────────────────────────────────────────────────────────────
def build_chunks(payload: bytes,
                 signing_key: nacl.signing.SigningKey,
                 flight_id: str,
                 session_id: str,
                 chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """
    Split payload into chunks. Each chunk gets:
      - packet_id   : FLT0001_a3f2c1_000042
      - seq         : 0-based integer
      - total       : total chunk count
      - ts_sent_ns  : nanosecond UTC timestamp at build time
      - crc32       : CRC of payload
      - signature   : Ed25519(session_id + seq + payload)
      - payload     : the data bytes
    """
    import zlib
    n     = (len(payload) + chunk_size - 1) // chunk_size
    now_s = time.time()
    chunks = []
    for seq in range(n):
        pload     = payload[seq * chunk_size : (seq + 1) * chunk_size]
        ts_ns     = time.time_ns()
        sign_data = session_id.encode() + struct.pack(">I", seq) + pload
        signature = signing_key.sign(sign_data).signature  # 64 bytes
        crc       = zlib.crc32(pload) & 0xFFFFFFFF
        chunks.append({
            "packet_id":    f"{flight_id}_{session_id}_{seq:06d}",
            "seq":          seq,
            "total":        n,
            "ts_sent_ns":   ts_ns,
            "ts_built_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(ts_ns // 1_000_000_000)
            ) + f".{ts_ns % 1_000_000_000:09d}Z",
            "session_id":   session_id,
            "flight_id":    flight_id,
            "crc32":        crc,
            "signature":    signature,
            "payload":      pload,
        })
    return chunks


def serialise_chunk(c: dict) -> bytes:
    """
    Wire format per chunk (all big-endian):
      [MAGIC 4B "C717"]
      [seq 4B][total 4B][ts_ns 8B][crc32 4B]
      [packet_id 32B padded ASCII]
      [session_id 8B padded ASCII]
      [flight_id 8B padded ASCII]
      [sig 64B]
      [payload_len 4B][payload]
    """
    pid  = c["packet_id"].encode()[:32].ljust(32, b"\x00")
    sid  = c["session_id"].encode()[:8].ljust(8, b"\x00")
    fid  = c["flight_id"].encode()[:8].ljust(8, b"\x00")
    hdr  = struct.pack(">4sIIQI", b"C717",
                       c["seq"], c["total"],
                       c["ts_sent_ns"], c["crc32"])
    return hdr + pid + sid + fid + c["signature"] + struct.pack(">I", len(c["payload"])) + c["payload"]


def deserialise_chunk(data: bytes) -> dict:
    """Parse a serialised chunk back into a dict."""
    import zlib
    magic, seq, total, ts_ns, crc = struct.unpack_from(">4sIIQI", data, 0)
    assert magic == b"C717", f"bad magic {magic!r}"
    off   = struct.calcsize(">4sIIQI")
    pid   = data[off:off+32].rstrip(b"\x00").decode()  ; off += 32
    sid   = data[off:off+8].rstrip(b"\x00").decode()   ; off += 8
    fid   = data[off:off+8].rstrip(b"\x00").decode()   ; off += 8
    sig   = data[off:off+64]                            ; off += 64
    plen  = struct.unpack_from(">I", data, off)[0]     ; off += 4
    pload = data[off:off+plen]
    actual_crc = zlib.crc32(pload) & 0xFFFFFFFF
    return {
        "packet_id":  pid,
        "seq":        seq,
        "total":      total,
        "ts_sent_ns": ts_ns,
        "crc32":      crc,
        "crc_ok":     actual_crc == crc,
        "session_id": sid,
        "flight_id":  fid,
        "signature":  sig,
        "payload":    pload,
    }


# ── Manifest ───────────────────────────────────────────────────────────────────
def build_manifest(chunks: list[dict],
                   telem_size: int,
                   audio_size: int,
                   audio_meta: dict,
                   pub_key_hex: str,
                   bundle_sha256: str,
                   session_id: str,
                   flight_id: str,
                   event_type: str = "DISTRESS") -> bytes:
    """
    Manifest is sent first. Receiver uses it to:
      - know total chunks expected
      - know audio format for reconstruction
      - verify public key matches the one it has on file
    """
    m = {
        "version":       2,
        "session_id":    session_id,
        "flight_id":     flight_id,
        "event_type":    event_type,
        "ts_created_ns": time.time_ns(),
        "ts_created_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks":  len(chunks),
        "chunk_size":    CHUNK_SIZE,
        "telem_bytes_raw":   telem_size,
        "audio_bytes_raw":   audio_size,
        "audio_meta":    audio_meta,
        "bundle_sha256": bundle_sha256,
        "pubkey_hex":    pub_key_hex,
        "packet_ids":    [c["packet_id"] for c in chunks],  # full list for tracking
    }
    return json.dumps(m, indent=2).encode()


# ── libp2p host ────────────────────────────────────────────────────────────────
def make_host(port: int):
    kp    = create_new_key_pair(secrets.token_bytes(32))
    noise = NoiseTransport(libp2p_keypair=kp, noise_privkey=kp.private_key, early_data=None)
    rm    = new_resource_manager(ResourceLimits(max_connections=32, max_streams=256, max_memory_mb=512))
    return new_host(key_pair=kp, sec_opt={NOISE_PROTOCOL_ID: noise},
                    resource_manager=rm, enable_quic=True), kp

def get_quic_addrs(host, port):
    addrs = get_available_interfaces(port, protocol="udp")
    result = []
    for a in addrs:
        s = str(a)
        if "/ip4/" in s:
            result.append(Multiaddr(s.replace("/tcp/", "/udp/") + "/quic-v1"))
    return result


# ── Per-chunk metric record ────────────────────────────────────────────────────
class ChunkLog:
    def __init__(self):
        self.records = []   # list of dicts, one per chunk

    def record(self, seq, packet_id, size, ts_sent_ns, ts_acked_ns,
               retransmits, lost, crc_ok_at_recv=None, sig_ok_at_recv=None):
        rtt_ns = (ts_acked_ns - ts_sent_ns) if ts_acked_ns and not lost else None
        self.records.append({
            "packet_id":       packet_id,
            "seq":             seq,
            "size_bytes":      size,
            "ts_sent_ns":      ts_sent_ns,
            "ts_sent_iso":     _ns_to_iso(ts_sent_ns),
            "ts_acked_ns":     ts_acked_ns,
            "ts_acked_iso":    _ns_to_iso(ts_acked_ns) if ts_acked_ns else None,
            "rtt_ns":          rtt_ns,
            "rtt_ms":          round(rtt_ns / 1e6, 3) if rtt_ns else None,
            "retransmits":     retransmits,
            "delivered":       not lost,
            "crc_ok_at_recv":  crc_ok_at_recv,
            "sig_ok_at_recv":  sig_ok_at_recv,
        })

    def summary(self) -> dict:
        total    = len(self.records)
        lost     = [r for r in self.records if not r["delivered"]]
        retxd    = [r for r in self.records if r["retransmits"] > 0]
        rtts_ms  = [r["rtt_ms"] for r in self.records if r["rtt_ms"] is not None]
        elapsed  = 0
        if self.records:
            elapsed = (max(r["ts_acked_ns"] or 0 for r in self.records)
                       - self.records[0]["ts_sent_ns"]) / 1e9

        def pct(lst, n): return round(sorted(lst)[int((pct/100)*n)], 3) if lst else None

        def percentile(lst, p):
            if not lst: return None
            idx = max(0, int(p/100 * len(lst)) - 1)
            return round(sorted(lst)[idx], 3)

        return {
            "total_chunks":          total,
            "delivered":             total - len(lost),
            "lost":                  len(lost),
            "loss_pct":              round(len(lost)/max(total,1)*100, 3),
            "retransmit_events":     sum(r["retransmits"] for r in self.records),
            "chunks_needing_retx":   len(retxd),
            "elapsed_sec":           round(elapsed, 3),
            "throughput_bps":        round(sum(r["size_bytes"] for r in self.records)*8 / max(elapsed,0.001)),
            "rtt_min_ms":            round(min(rtts_ms), 3) if rtts_ms else None,
            "rtt_max_ms":            round(max(rtts_ms), 3) if rtts_ms else None,
            "rtt_avg_ms":            round(sum(rtts_ms)/len(rtts_ms), 3) if rtts_ms else None,
            "rtt_p50_ms":            percentile(rtts_ms, 50),
            "rtt_p95_ms":            percentile(rtts_ms, 95),
            "rtt_p99_ms":            percentile(rtts_ms, 99),
            "lost_packet_ids":       [r["packet_id"] for r in lost],
            "retransmit_packet_ids": [r["packet_id"] for r in retxd],
        }

    def print_summary(self, session_id: str):
        s = self.summary()
        w = 62
        print(f"\n{'═'*w}")
        print(f"  SESSION METRICS  id={session_id}")
        print(f"{'═'*w}")
        print(f"  Chunks total     : {s['total_chunks']}")
        print(f"  Delivered        : {s['delivered']}  ✓")
        print(f"  Lost             : {s['lost']}  {'✓' if s['lost']==0 else '✗'}")
        print(f"  Loss %           : {s['loss_pct']:.3f}%")
        print(f"  Retransmit events: {s['retransmit_events']}")
        print(f"  Elapsed          : {s['elapsed_sec']} s")
        print(f"  Throughput       : {s['throughput_bps']:,} bps")
        if s['rtt_avg_ms']:
            print(f"  RTT avg/p50/p95/p99: "
                  f"{s['rtt_avg_ms']} / {s['rtt_p50_ms']} / "
                  f"{s['rtt_p95_ms']} / {s['rtt_p99_ms']} ms")
        if s['lost_packet_ids']:
            print(f"  Lost IDs         : {s['lost_packet_ids'][:5]}")
        if s['retransmit_packet_ids']:
            print(f"  Retransmit IDs   : {s['retransmit_packet_ids'][:5]}")
        print(f"{'═'*w}")


def _ns_to_iso(ns: int) -> str:
    if not ns: return None
    sec = ns // 1_000_000_000
    frac = ns % 1_000_000_000
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(sec)) + f".{frac:09d}Z"


# ── Main send coroutine ────────────────────────────────────────────────────────
async def run_sender(manifest_bytes: bytes,
                     chunks: list[dict],
                     peer_addr: str,
                     port: int,
                     session_id: str,
                     flight_id: str):

    log  = ChunkLog()
    host, _ = make_host(port)
    addrs   = get_quic_addrs(host, port)

    ack_arrived = trio.Event()
    ack_holder  = {"stream": None}

    async def _on_ack(stream):
        ack_holder["stream"] = stream
        ack_arrived.set()

    host.set_stream_handler(PROTO_ACK,  _on_ack)
    host.set_stream_handler(PROTO_DATA, lambda s: None)

    async with host.run(listen_addrs=addrs):
        print(f"\n[SENDER] Local peer: {host.get_id().to_string()[:28]}...")

        # ── Connect with retry ────────────────────────────────────────────
        peer_info = info_from_p2p_addr(Multiaddr(peer_addr))
        for attempt in range(1, CONNECT_RETRY + 1):
            try:
                print(f"[SENDER] Connecting (attempt {attempt})...")
                await host.connect(peer_info)
                print(f"[SENDER] Connected ✓")
                break
            except Exception as e:
                if attempt == CONNECT_RETRY:
                    raise RuntimeError(f"Cannot connect: {e}")
                await trio.sleep(2 ** attempt)

        data_stream = await host.new_stream(peer_info.peer_id, [PROTO_DATA])

        try:
            # ── Send manifest ─────────────────────────────────────────────
            print(f"[SENDER] Sending manifest ({len(manifest_bytes):,} B)...")
            await send_msg(data_stream, manifest_bytes)

            # ── Wait for ACK channel ──────────────────────────────────────
            print(f"[SENDER] Waiting for ACK channel...")
            with trio.move_on_after(30) as cs:
                await ack_arrived.wait()
            if cs.cancelled_caught:
                raise RuntimeError("ACK channel timeout")
            ack_stream = ack_holder["stream"]
            with trio.move_on_after(30) as cs:
                ready = await recv_msg(ack_stream)
            if cs.cancelled_caught or ready != b"READY":
                raise RuntimeError(f"Bad READY: {ready!r}")
            print(f"[SENDER] ACK channel ready ✓")

            # ── Send chunks ───────────────────────────────────────────────
            n = len(chunks)
            print(f"[SENDER] Sending {n} chunks...")

            for c in chunks:
                wire     = serialise_chunk(c)
                sent_at  = time.time_ns()
                retx     = 0
                delivered = False

                for attempt in range(CONNECT_RETRY + 1):
                    await send_msg(data_stream, wire)

                    with trio.move_on_after(ACK_TIMEOUT) as cs:
                        ack = await recv_msg(ack_stream)

                    if cs.cancelled_caught:
                        retx += 1
                        print(f"  [SENDER] seq={c['seq']} timeout → retransmit #{retx}")
                        sent_at = time.time_ns()
                        continue

                    # Parse ACK — receiver sends back JSON with metrics
                    try:
                        ack_data    = json.loads(ack.decode())
                        acked_at    = time.time_ns()
                        crc_ok      = ack_data.get("crc_ok", None)
                        sig_ok      = ack_data.get("sig_ok", None)
                        recv_seq    = ack_data.get("seq", -1)
                    except Exception:
                        # Fallback: plain text ACK:<seq>
                        acked_at = time.time_ns()
                        crc_ok   = None
                        sig_ok   = None
                        recv_seq = c["seq"]

                    if recv_seq == c["seq"]:
                        log.record(c["seq"], c["packet_id"],
                                   len(wire), sent_at, acked_at,
                                   retx, False, crc_ok, sig_ok)
                        delivered = True
                        rtt_ms = (acked_at - sent_at) / 1e6

                        # Progress: every 10 chunks or first/last
                        if c["seq"] % 10 == 0 or c["seq"] == n-1 or retx > 0:
                            crc_s = f"crc={'✓' if crc_ok else '✗'}" if crc_ok is not None else ""
                            sig_s = f"sig={'✓' if sig_ok else '✗'}" if sig_ok is not None else ""
                            print(f"  [SENDER] {c['seq']+1:4d}/{n}  "
                                  f"pid={c['packet_id']}  "
                                  f"RTT={rtt_ms:.2f}ms  {crc_s} {sig_s}"
                                  + (f"  RETX={retx}" if retx else ""))
                        break
                    else:
                        retx += 1

                if not delivered:
                    log.record(c["seq"], c["packet_id"],
                               len(wire), sent_at, None, retx, True)
                    print(f"  [SENDER] {c['packet_id']} LOST after {retx} retransmits")

            # ── Signal done ───────────────────────────────────────────────
            await send_msg(data_stream, b"DONE")
            with trio.move_on_after(60) as cs:
                final = await recv_msg(ack_stream)
            status = final.decode() if not cs.cancelled_caught else "TIMEOUT"
            print(f"[SENDER] Final status: {status!r}")

            await data_stream.close_write()

        except Exception as e:
            print(f"[SENDER] Error: {e}")
            import traceback; traceback.print_exc()
            try: await data_stream.reset()
            except Exception: pass

    log.print_summary(session_id)

    # Save metrics JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = f"{OUTPUT_DIR}/{flight_id}_{session_id}_metrics.json"
    with open(metrics_path, "w") as f:
        data_out = {"summary": log.summary(), "chunks": log.records}
        json.dump(data_out, f, indent=2)
    print(f"[SENDER] Metrics saved: {metrics_path}")
    return log


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="ARINC 717 QUIC Sender")
    p.add_argument("--telem",          required=True,      help="Telemetry .bin file")
    p.add_argument("--audio",          default=None,       help="Audio file (.wav / .bin / .pcm)")
    p.add_argument("--audio-rate",     type=int, default=16000, help="Sample rate for raw .bin audio")
    p.add_argument("--audio-bits",     type=int, default=16,    help="Bit depth for raw .bin audio")
    p.add_argument("--audio-channels", type=int, default=1,     help="Channels for raw .bin audio")
    p.add_argument("--audio-last-sec", type=int, default=None,  help="Keep last N sec of audio")
    p.add_argument("--peer",           required=True,      help="Receiver multiaddr")
    p.add_argument("--port",           type=int, default=8001,  help="Local sender port")
    p.add_argument("--flight",         default="FLT0001",  help="Flight ID")
    p.add_argument("--key",            default=PRIVATE_KEY_FILE, help="Private key file")
    p.add_argument("--event",          default="DISTRESS", help="Event type label")
    p.add_argument("--chunk-size",     type=int, default=CHUNK_SIZE, help="Bytes per chunk")
    args = p.parse_args()

    session_id = secrets.token_hex(4)
    print(f"\n{'█'*64}")
    print(f"  SENDER  flight={args.flight}  session={session_id}")
    print(f"{'█'*64}")

    # ── Load private key ──────────────────────────────────────────────────
    print(f"\n[KEY] Loading private key: {args.key}")
    signing_key = load_private_key(args.key)
    pub_hex     = signing_key.verify_key.encode(nacl.encoding.HexEncoder).decode()
    print(f"[KEY] Public key: {pub_hex[:32]}...")

    # ── Load telemetry ────────────────────────────────────────────────────
    print(f"\n[TELEM] Loading: {args.telem}")
    telem_raw = open(args.telem, "rb").read()
    print(f"[TELEM] {len(telem_raw):,} bytes")

    # ── Load audio (optional) ─────────────────────────────────────────────
    audio_raw  = b""
    audio_meta = {}
    if args.audio:
        audio_raw, audio_meta = load_audio_file(
            args.audio,
            sample_rate=args.audio_rate,
            channels=args.audio_channels,
            bit_depth=args.audio_bits,
            last_n_sec=args.audio_last_sec,
        )
    else:
        print("[AUDIO] No audio file provided — sending telemetry only")

    # ── Pack payload: [4B telem_len][telem][4B audio_len][audio] ─────────
    import struct as _s
    payload_raw = (_s.pack(">I", len(telem_raw)) + telem_raw
                   + _s.pack(">I", len(audio_raw)) + audio_raw)

    # ── Compress ──────────────────────────────────────────────────────────
    print(f"\n[COMPRESS] Compressing {len(payload_raw):,} bytes...")
    t0    = time.time()
    comp  = compress(payload_raw)
    dt    = time.time() - t0
    cr    = len(payload_raw) / max(len(comp), 1)
    sha   = hashlib.sha256(comp).hexdigest()
    print(f"[COMPRESS] {len(payload_raw):,} → {len(comp):,} bytes  "
          f"CR={cr:.2f}×  time={dt*1000:.0f}ms")
    print(f"[COMPRESS] SHA256: {sha}")

    # ── Build chunks + sign ───────────────────────────────────────────────
    print(f"\n[CHUNK] Splitting into {args.chunk_size:,}-byte chunks...")
    chunks = build_chunks(comp, signing_key, args.flight, session_id, args.chunk_size)
    print(f"[CHUNK] {len(chunks)} chunks  "
          f"({len(chunks)*args.chunk_size//1024} KB wire approx)")

    # ── Build manifest ─────────────────────────────────────────────────────
    manifest = build_manifest(chunks, len(telem_raw), len(audio_raw),
                              audio_meta, pub_hex, sha,
                              session_id, args.flight, args.event)

    # ── Send ──────────────────────────────────────────────────────────────
    trio.run(run_sender, manifest, chunks, args.peer,
             args.port, session_id, args.flight)


if __name__ == "__main__":
    main()
