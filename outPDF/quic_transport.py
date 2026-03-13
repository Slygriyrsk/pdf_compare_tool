"""
MODULE 4 — quic_transport.py
=============================
libp2p QUIC sender and receiver for ARINC 717 bundles.

Design decisions:
  - Each chunk is a separate stream.write() call with 4-byte length prefix
    (framing) because QUIC is a byte stream, not a message protocol.
  - Manifest is sent first; receiver parses it to know total chunks.
  - Per-chunk ACK: receiver sends "ACK:<seq>" back on the same stream
    before the sender moves to the next chunk. This gives us per-chunk
    RTT measurement.
  - Missing chunks after all sends: sender re-sends those only (DTN retransmit).
  - TransportMetrics tracks every packet: seq, size, RTT, retransmit count,
    timestamp, lost flag.

Limitations of this prototype:
  - One stream per session (simpler, avoids QUIC concurrent stream negotiation)
  - No multi-path (single QUIC connection)
  - No bandwidth shaping / pacing (QUIC itself does congestion control)
"""

import os
import json
import struct
import time
import secrets
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict

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

PROTOCOL       = "/arinc717/1.0.0"
FRAME_TIMEOUT  = 30     # sec — per-chunk send/receive timeout
CONNECT_RETRY  = 3
RECV_SAVE_DIR  = "./received_bundles"


# ── Wire framing helpers ───────────────────────────────────────────────────────
# QUIC is a raw byte stream: no message boundaries.
# We prefix every message with a 4-byte big-endian length.

async def send_msg(stream, data: bytes) -> None:
    """Send length-prefixed message."""
    await stream.write(struct.pack(">I", len(data)) + data)


async def recv_msg(stream) -> bytes:
    """Receive one length-prefixed message. Loops until all bytes arrive."""
    hdr = b""
    while len(hdr) < 4:
        chunk = await stream.read(4 - len(hdr))
        if not chunk:
            raise EOFError("Stream closed before header complete")
        hdr += chunk
    length = struct.unpack(">I", hdr)[0]

    data = b""
    while len(data) < length:
        chunk = await stream.read(length - len(data))
        if not chunk:
            raise EOFError(f"Truncated: got {len(data)}/{length} bytes")
        data += chunk
    return data


# ── Metrics ────────────────────────────────────────────────────────────────────
@dataclass
class ChunkMetric:
    seq:          int
    size_bytes:   int
    sent_at:      float
    acked_at:     Optional[float] = None
    retransmits:  int = 0
    lost:         bool = False

    @property
    def rtt_ms(self) -> Optional[float]:
        if self.acked_at and not self.lost:
            return (self.acked_at - self.sent_at) * 1000
        return None


@dataclass
class TransportMetrics:
    session_id:    str = field(default_factory=lambda: secrets.token_hex(4))
    flight_id:     str = ""
    start_time:    float = field(default_factory=time.time)
    end_time:      Optional[float] = None
    total_bytes:   int = 0
    chunks:        List[ChunkMetric] = field(default_factory=list)

    def record(self, seq, size, sent_at, acked_at, retransmits, lost):
        self.chunks.append(ChunkMetric(seq, size, sent_at, acked_at,
                                       retransmits, lost))

    def report(self) -> dict:
        rtts      = [c.rtt_ms for c in self.chunks if c.rtt_ms is not None]
        lost      = [c for c in self.chunks if c.lost]
        retxd     = [c for c in self.chunks if c.retransmits > 0]
        elapsed   = (self.end_time or time.time()) - self.start_time
        tput_bps  = self.total_bytes * 8 / max(elapsed, 0.001)

        r = {
            "session_id":       self.session_id,
            "flight_id":        self.flight_id,
            "elapsed_sec":      round(elapsed, 2),
            "total_bytes_sent": self.total_bytes,
            "throughput_bps":   round(tput_bps, 1),
            "chunks_sent":      len(self.chunks),
            "chunks_acked":     len(rtts),
            "chunks_lost":      len(lost),
            "loss_pct":         round(len(lost) / max(len(self.chunks), 1) * 100, 2),
            "retransmissions":  sum(c.retransmits for c in self.chunks),
            "rtt_avg_ms":       round(sum(rtts) / len(rtts), 2) if rtts else None,
            "rtt_min_ms":       round(min(rtts), 2) if rtts else None,
            "rtt_max_ms":       round(max(rtts), 2) if rtts else None,
            "rtt_p99_ms":       round(sorted(rtts)[int(0.99 * len(rtts))], 2) if len(rtts) > 1 else None,
            "lost_seqs":        [c.seq for c in lost],
            "retransmit_seqs":  [c.seq for c in retxd],
        }
        return r

    def print_report(self):
        r = self.report()
        print(f"\n{'='*60}")
        print(f"  TRANSPORT METRICS  session={r['session_id']}")
        print(f"{'='*60}")
        print(f"  Flight ID         : {r['flight_id']}")
        print(f"  Elapsed           : {r['elapsed_sec']} s")
        print(f"  Bytes sent        : {r['total_bytes_sent']:,}")
        print(f"  Throughput        : {r['throughput_bps']:.0f} bps")
        print(f"  Chunks sent       : {r['chunks_sent']}")
        print(f"  Chunks acked      : {r['chunks_acked']}")
        print(f"  Chunks lost       : {r['chunks_lost']}")
        print(f"  Loss rate         : {r['loss_pct']:.2f}%")
        print(f"  Retransmissions   : {r['retransmissions']}")
        if r['rtt_avg_ms']:
            print(f"  RTT avg           : {r['rtt_avg_ms']} ms")
            print(f"  RTT min           : {r['rtt_min_ms']} ms")
            print(f"  RTT max           : {r['rtt_max_ms']} ms")
            print(f"  RTT p99           : {r['rtt_p99_ms']} ms")
        if r['lost_seqs']:
            print(f"  Lost chunks       : {r['lost_seqs'][:20]}")
        if r['retransmit_seqs']:
            print(f"  Retransmit chunks : {r['retransmit_seqs'][:20]}")
        print(f"{'='*60}")


# ── QUIC host factory ──────────────────────────────────────────────────────────
def make_host(port: int, enable_quic: bool = True):
    """Create a libp2p host with Noise security and QUIC transport."""
    key_pair = create_new_key_pair(secrets.token_bytes(32))
    noise    = NoiseTransport(
        libp2p_keypair=key_pair,
        noise_privkey=key_pair.private_key,
        early_data=None,
    )
    limits = ResourceLimits(max_connections=64, max_streams=256, max_memory_mb=128)
    rm     = new_resource_manager(limits=limits)
    host   = new_host(
        key_pair=key_pair,
        sec_opt={NOISE_PROTOCOL_ID: noise},
        resource_manager=rm,
        enable_quic=enable_quic,
    )
    return host, key_pair


def quic_addrs(host, port: int) -> List[Multiaddr]:
    """Return IPv4 QUIC-v1 listen addresses for a given port."""
    addrs = get_available_interfaces(port, protocol="udp")
    return [
        Multiaddr(str(a).replace("/tcp/", "/udp/") + "/quic-v1")
        for a in addrs if "/ip4/" in str(a)
    ]


# ── SENDER ────────────────────────────────────────────────────────────────────
async def send_bundle(manifest_chunk: bytes,
                      data_chunks: List[bytes],
                      peer_addr: str,
                      port: int = 8001,
                      flight_id: str = "FLT0001") -> TransportMetrics:
    """
    Send manifest + all data chunks over QUIC.
    Per-chunk ACK loop with retransmit on timeout.

    Flow:
      1. Connect to peer
      2. Send manifest, wait for "MANIFEST_OK"
      3. For each chunk:
           send chunk → wait for "ACK:<seq>" → record RTT
           on timeout → retransmit (up to CONNECT_RETRY times)
      4. Sender signals "DONE", waits for "RECV_COMPLETE"
      5. Print metrics
    """
    metrics          = TransportMetrics(flight_id=flight_id)
    metrics.total_bytes = sum(len(c) for c in data_chunks) + len(manifest_chunk)

    host, _ = make_host(port)
    addrs   = quic_addrs(host, port)

    # Register protocol handler (sender side doesn't handle inbound, but required)
    host.set_stream_handler(PROTOCOL, lambda s: None)

    async with host.run(listen_addrs=addrs):
        peer_id_str = host.get_id().to_string()
        print(f"[SENDER] Started  port={port}  peer={peer_id_str[:20]}...")

        # ── Connect with retry ────────────────────────────────────────────────
        peer_info = info_from_p2p_addr(Multiaddr(peer_addr))
        for attempt in range(1, CONNECT_RETRY + 1):
            try:
                print(f"[SENDER] Connecting (attempt {attempt})...")
                await host.connect(peer_info)
                print(f"[SENDER] Connected to {peer_info.peer_id.to_string()[:20]}...")
                break
            except Exception as e:
                print(f"[SENDER] Connect failed: {e}")
                if attempt == CONNECT_RETRY:
                    raise RuntimeError(f"Cannot connect after {CONNECT_RETRY} attempts")
                await trio.sleep(2 ** attempt)

        stream = await host.new_stream(peer_info.peer_id, [PROTOCOL])
        print(f"[SENDER] Stream open  protocol={PROTOCOL}")

        try:
            # ── Step 1: send manifest ─────────────────────────────────────────
            print(f"[SENDER] Sending manifest ({len(manifest_chunk)}B)...")
            await send_msg(stream, manifest_chunk)
            with trio.move_on_after(FRAME_TIMEOUT) as cs:
                ack = await recv_msg(stream)
            if cs.cancelled_caught or ack != b"MANIFEST_OK":
                raise RuntimeError(f"Manifest not acknowledged: {ack!r}")
            print(f"[SENDER] Manifest acknowledged")

            # ── Step 2: send each data chunk ──────────────────────────────────
            n = len(data_chunks)
            for seq, chunk in enumerate(data_chunks):
                sent = False
                retx = 0
                for attempt in range(CONNECT_RETRY + 1):
                    sent_at = time.time()
                    await send_msg(stream, chunk)

                    with trio.move_on_after(FRAME_TIMEOUT) as cs:
                        ack = await recv_msg(stream)

                    if cs.cancelled_caught:
                        print(f"  [SENDER] seq={seq} timeout, retransmit {retx+1}")
                        retx += 1
                        continue

                    expected = f"ACK:{seq}".encode()
                    if ack == expected:
                        acked_at = time.time()
                        rtt      = (acked_at - sent_at) * 1000
                        metrics.record(seq, len(chunk), sent_at, acked_at, retx, False)
                        sent = True

                        # Progress every 50 chunks or on retransmit or last chunk
                        if seq % 50 == 0 or retx > 0 or seq == n - 1:
                            pct = (seq + 1) / n * 100
                            print(f"  [SENDER] chunk {seq+1:4d}/{n}  "
                                  f"{pct:5.1f}%  RTT={rtt:.1f}ms"
                                  + (f"  RETX={retx}" if retx else ""))
                        break
                    else:
                        print(f"  [SENDER] seq={seq} bad ACK: {ack!r}")
                        retx += 1

                if not sent:
                    metrics.record(seq, len(chunk), sent_at, None, retx, True)
                    print(f"  [SENDER] seq={seq} LOST after {retx} retransmits")

            # ── Step 3: signal done ───────────────────────────────────────────
            await send_msg(stream, b"DONE")
            with trio.move_on_after(FRAME_TIMEOUT):
                final = await recv_msg(stream)
            print(f"[SENDER] Final status: {final.decode()!r}")

            await stream.close_write()

        except Exception as e:
            print(f"[SENDER] Error: {e}")
            try:
                await stream.reset()
            except Exception:
                pass

        finally:
            metrics.end_time = time.time()

    metrics.print_report()
    return metrics


# ── RECEIVER ──────────────────────────────────────────────────────────────────
class BundleReceiver:
    """
    Stateful receiver: collects chunks, tracks missing ones, reassembles.
    Designed to handle partial delivery (DTN store-and-forward concept).
    """

    def __init__(self):
        self.manifest:      Optional[dict]         = None
        self.received:      Dict[int, bytes]        = {}   # seq → raw chunk
        self.missing:       set                     = set()
        self.total_chunks:  int                     = 0
        self.bundle_id:     str                     = ""
        os.makedirs(RECV_SAVE_DIR, exist_ok=True)

    def got_manifest(self, raw_chunk: bytes):
        from packet_framer import parse_chunk
        c = parse_chunk(raw_chunk)
        if c and c["is_manifest"]:
            self.manifest     = json.loads(c["payload"])
            self.total_chunks = self.manifest["n_chunks"]
            self.bundle_id    = self.manifest["bundle_id"]
            self.missing      = set(range(self.total_chunks))
            print(f"[RECV] Manifest: {self.total_chunks} chunks expected  "
                  f"bundle={self.bundle_id[:16]}...")

    def got_chunk(self, raw_chunk: bytes) -> int:
        """Store chunk. Returns seq number or -1 on error."""
        from packet_framer import parse_chunk
        c = parse_chunk(raw_chunk)
        if not c or c["is_manifest"]:
            return -1
        seq = c["seq"]
        self.received[seq] = raw_chunk
        self.missing.discard(seq)
        return seq

    def is_complete(self) -> bool:
        return self.total_chunks > 0 and len(self.missing) == 0

    def reassemble_and_save(self, verifier) -> Optional[str]:
        """
        Reassemble all received chunks into bundle file.
        Returns output file path or None on failure.
        """
        from packet_framer import parse_chunk, reassemble, MANIFEST_SEQ
        import nacl.signing

        parsed = [parse_chunk(r) for r in self.received.values() if r]
        result, report = reassemble(parsed, verifier)
        print(f"[RECV] Reassemble: {report}")

        if report["missing_seqs"]:
            print(f"[RECV] WARNING: {len(report['missing_seqs'])} missing chunks")

        # Verify final SHA256
        if self.manifest:
            expected_sha = self.manifest.get("sha256", "")
            actual_sha   = hashlib.sha256(result).hexdigest()
            if actual_sha != expected_sha:
                print(f"[RECV] ERROR: SHA256 mismatch!  "
                      f"expected={expected_sha[:16]} got={actual_sha[:16]}")
                return None
            print(f"[RECV] SHA256 verified ✓")

        # Save to disk
        fname   = f"{RECV_SAVE_DIR}/{self.bundle_id[:8]}_{int(time.time())}.a717"
        with open(fname, "wb") as f:
            f.write(result)
        print(f"[RECV] Saved bundle: {fname}  ({len(result):,} bytes)")
        return fname


async def _handle_stream(stream, receiver: BundleReceiver, verifier):
    """
    Handle one incoming QUIC stream:
    1. Receive manifest → ACK "MANIFEST_OK"
    2. Receive chunks → ACK each with "ACK:<seq>"
    3. Receive "DONE" → reassemble → send "RECV_COMPLETE" or "RECV_PARTIAL"
    """
    print(f"[RECV] Incoming stream  protocol={stream.get_protocol()}")
    chunks_received = 0

    try:
        # Manifest
        with trio.move_on_after(FRAME_TIMEOUT) as cs:
            raw_mf = await recv_msg(stream)
        if cs.cancelled_caught:
            print("[RECV] Timeout waiting for manifest")
            return
        receiver.got_manifest(raw_mf)
        await send_msg(stream, b"MANIFEST_OK")

        # Data chunks
        while True:
            with trio.move_on_after(FRAME_TIMEOUT) as cs:
                raw = await recv_msg(stream)
            if cs.cancelled_caught:
                print("[RECV] Timeout waiting for chunk")
                break
            if raw == b"DONE":
                break

            seq = receiver.got_chunk(raw)
            if seq >= 0:
                chunks_received += 1
                await send_msg(stream, f"ACK:{seq}".encode())
                if chunks_received % 50 == 0:
                    total = receiver.total_chunks
                    pct   = chunks_received / max(total, 1) * 100
                    print(f"  [RECV] {chunks_received}/{total}  {pct:.1f}%  "
                          f"missing={len(receiver.missing)}")
            else:
                print(f"[RECV] Bad chunk received, skipping")

        # Reassemble
        print(f"[RECV] All chunks received ({chunks_received}). Reassembling...")
        fname = receiver.reassemble_and_save(verifier)
        status = b"RECV_COMPLETE" if fname else b"RECV_PARTIAL"
        await send_msg(stream, status)

        await stream.close_write()

    except Exception as e:
        print(f"[RECV] Stream error: {e}")
        try:
            await stream.reset()
        except Exception:
            pass
    finally:
        try:
            await stream.close()
        except Exception:
            pass


async def run_receiver(port: int = 8000,
                       verifier=None,
                       max_bundles: int = 1):
    """
    Start QUIC receiver, handle up to max_bundles incoming sessions.
    Prints the bootstrap address that sender needs.
    """
    from packet_framer import Verifier as V

    host, _ = make_host(port)
    addrs   = quic_addrs(host, port)

    bundles_done = 0
    done_event   = trio.Event()

    async def handle(stream):
        nonlocal bundles_done
        receiver = BundleReceiver()
        # Use verifier passed in (contains airplane's public key)
        v = verifier
        await _handle_stream(stream, receiver, v)
        bundles_done += 1
        if bundles_done >= max_bundles:
            done_event.set()

    # Register handler BEFORE host.run()
    host.set_stream_handler(PROTOCOL, handle)

    async with host.run(listen_addrs=addrs):
        pid  = host.get_id().to_string()
        print(f"\n[RECV] libp2p QUIC receiver started")
        print(f"[RECV] Peer ID: {pid}")
        for a in addrs:
            print(f"[RECV] Listening: {a}/p2p/{pid}")

        # Wait until we've received the expected number of bundles
        await done_event.wait()
        print(f"[RECV] Received {bundles_done} bundle(s). Shutting down.")


# ── Entry points ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "receiver":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        print(f"[MAIN] Starting receiver on port {port}")
        trio.run(run_receiver, port)

    elif mode == "sender":
        if len(sys.argv) < 3:
            print("Usage: python quic_transport.py sender <peer_multiaddr>")
            sys.exit(1)
        peer_addr = sys.argv[2]

        # Quick test bundle
        sys.path.insert(0, os.path.dirname(__file__))
        from compressor import compress_payload, pack_bundle, simulate_cvr_audio, EVT_DISTRESS
        from arinc_generator import generate_telemetry
        from packet_framer import build_chunks, Signer

        raw, _, _, _ = generate_telemetry(60)   # 1 min for quick test
        cvr = simulate_cvr_audio(60)
        comp, _ = compress_payload(raw, cvr)
        bundle  = pack_bundle(comp, event_type=EVT_DISTRESS, pre_event_sec=60)
        signer  = Signer()
        mf, chunks = build_chunks(bundle, signer)

        trio.run(send_bundle, mf, chunks, peer_addr)

    else:
        print("Usage: python quic_transport.py [receiver|sender] [args]")
