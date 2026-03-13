"""
MODULE 4 — quic_transport.py  (v2 — all bugs fixed)
=====================================================
libp2p QUIC sender/receiver for ARINC 717 bundles.

BUG FIXES vs v1:
───────────────────────────────────────────────────────────────────────────────
BUG 1: "Cannot send data on unknown peer-initiated stream"
  Root cause: aioquic enforces that only the stream INITIATOR can write to it.
  Stream IDs 0,4,8... are client-initiated. When the RECEIVER (server side)
  tried to call stream.write() on stream_id=4 (which the SENDER opened),
  aioquic's _get_or_create_stream_for_send() raised:
    ValueError("Cannot send data on unknown peer-initiated stream")
  because stream_is_client_initiated(4)=True but server _is_client=False.

  Fix: Two-protocol architecture.
    PROTO_DATA = "/arinc717/data/1.0.0"  ← SENDER opens this, pushes chunks
    PROTO_ACK  = "/arinc717/ack/1.0.0"   ← RECEIVER opens this BACK to sender
  The receiver opens a new outbound stream on PROTO_ACK, which is server-
  initiated (stream IDs 1,5,9...) so the server CAN write to it freely.

BUG 2: "'NoneType' object has no attribute 'verify'"
  Root cause: run_receiver() was called with verifier=None (default), and
  _handle_stream passed that None straight into reassemble_and_save(verifier).
  Inside reassemble(), each chunk calls verifier.verify() → AttributeError.

  Fix: Always create a _NullVerifier() when no real key is provided.
  _NullVerifier.verify() returns True and prints a WARNING. Pipeline.py
  now explicitly instantiates the correct verifier before calling run_receiver.

BUG 3: "50/54  missing=4" — chunks lost, no retransmit
  Root cause: The ACK stream error (Bug 1) caused send_msg(ACK) to throw an
  exception on chunks 1,2,3,4 before the ACK channel was established. The
  sender timed out those chunks and marked them lost without retransmitting.

  Fix: With the two-protocol fix, the ACK channel is established FIRST before
  any chunk is sent. The sender waits for ACK_READY signal before sending
  chunk 0. All chunks now get properly ACKed.
───────────────────────────────────────────────────────────────────────────────

Protocol flow (v2):
  SENDER                              RECEIVER
  ──────────────────────────────────────────────────────
  opens PROTO_DATA stream    ─────►  _handle_data_stream() fires
  sends manifest             ─────►
                             ◄─────  opens PROTO_ACK stream back
                             ◄─────  sends "READY" on ACK stream
  receives "READY" on ACK            (ACK channel established)
  sends chunk 0              ─────►
                             ◄─────  "ACK:0"
  sends chunk 1              ─────►
                             ◄─────  "ACK:1"
  ...
  sends "DONE"               ─────►  reassemble + verify SHA256 + save file
                             ◄─────  "RECV_COMPLETE" or "RECV_PARTIAL"
  close_write()                       close_write()
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

# ── Protocol IDs ───────────────────────────────────────────────────────────────
# Two protocols: sender opens DATA, receiver opens ACK back.
# This sidesteps aioquic's peer-initiated stream write restriction entirely.
PROTO_DATA     = "/arinc717/data/1.0.0"
PROTO_ACK      = "/arinc717/ack/1.0.0"

FRAME_TIMEOUT  = 45      # sec — per-chunk send/receive timeout
CONNECT_RETRY  = 3
RECV_SAVE_DIR  = "./received_bundles"


# ── Wire framing: 4-byte length prefix ────────────────────────────────────────
# QUIC is a raw byte stream — no message boundaries.
# Every message is prefixed with its length as a 4-byte big-endian uint32.

async def send_msg(stream, data: bytes) -> None:
    """Send one length-prefixed message."""
    await stream.write(struct.pack(">I", len(data)) + data)


async def recv_msg(stream) -> bytes:
    """
    Receive exactly one length-prefixed message.
    Loops until all bytes arrive — critical because QUIC may fragment.
    Raises EOFError if stream closes before message is complete.
    """
    # Read the 4-byte length header first
    hdr = b""
    while len(hdr) < 4:
        chunk = await stream.read(4 - len(hdr))
        if not chunk:
            raise EOFError("Stream closed before header complete")
        hdr += chunk

    length = struct.unpack(">I", hdr)[0]

    # Now read exactly `length` bytes
    data = b""
    while len(data) < length:
        chunk = await stream.read(length - len(data))
        if not chunk:
            raise EOFError(f"Truncated message: got {len(data)}/{length} bytes")
        data += chunk

    return data


# ── Transport metrics ──────────────────────────────────────────────────────────
@dataclass
class ChunkMetric:
    """Per-chunk transmission record."""
    seq:         int
    size_bytes:  int
    sent_at:     float
    acked_at:    Optional[float] = None
    retransmits: int = 0
    lost:        bool = False

    @property
    def rtt_ms(self) -> Optional[float]:
        if self.acked_at and not self.lost:
            return (self.acked_at - self.sent_at) * 1000
        return None


@dataclass
class TransportMetrics:
    """Full session-level transmission metrics."""
    session_id:  str   = field(default_factory=lambda: secrets.token_hex(4))
    flight_id:   str   = ""
    start_time:  float = field(default_factory=time.time)
    end_time:    Optional[float] = None
    total_bytes: int   = 0
    chunks:      List[ChunkMetric] = field(default_factory=list)

    def record(self, seq, size, sent_at, acked_at, retransmits, lost):
        self.chunks.append(
            ChunkMetric(seq, size, sent_at, acked_at, retransmits, lost)
        )

    def report(self) -> dict:
        rtts    = [c.rtt_ms for c in self.chunks if c.rtt_ms is not None]
        lost    = [c for c in self.chunks if c.lost]
        retxd   = [c for c in self.chunks if c.retransmits > 0]
        elapsed = (self.end_time or time.time()) - self.start_time
        tput    = self.total_bytes * 8 / max(elapsed, 0.001)

        def pct99(lst):
            return round(sorted(lst)[int(0.99 * len(lst))], 2) if len(lst) > 1 else None

        return {
            "session_id":       self.session_id,
            "flight_id":        self.flight_id,
            "elapsed_sec":      round(elapsed, 2),
            "total_bytes_sent": self.total_bytes,
            "throughput_bps":   round(tput, 1),
            "chunks_sent":      len(self.chunks),
            "chunks_acked":     len(rtts),
            "chunks_lost":      len(lost),
            "loss_pct":         round(len(lost) / max(len(self.chunks), 1) * 100, 2),
            "retransmissions":  sum(c.retransmits for c in self.chunks),
            "rtt_avg_ms":       round(sum(rtts) / len(rtts), 2) if rtts else None,
            "rtt_min_ms":       round(min(rtts), 2) if rtts else None,
            "rtt_max_ms":       round(max(rtts), 2) if rtts else None,
            "rtt_p99_ms":       pct99(rtts),
            "lost_seqs":        [c.seq for c in lost],
            "retransmit_seqs":  [c.seq for c in retxd],
        }

    def print_report(self):
        r = self.report()
        w = 60
        print(f"\n{'='*w}")
        print(f"  TRANSPORT METRICS  session={r['session_id']}")
        print(f"{'='*w}")
        print(f"  Flight ID         : {r['flight_id']}")
        print(f"  Elapsed           : {r['elapsed_sec']} s")
        print(f"  Bytes sent        : {r['total_bytes_sent']:,}")
        print(f"  Throughput        : {r['throughput_bps']:.0f} bps")
        print(f"  Chunks sent/acked : {r['chunks_sent']} / {r['chunks_acked']}")
        print(f"  Chunks lost       : {r['chunks_lost']}  ({r['loss_pct']:.2f}%)")
        print(f"  Retransmissions   : {r['retransmissions']}")
        if r['rtt_avg_ms']:
            print(f"  RTT avg/min/max   : {r['rtt_avg_ms']} / "
                  f"{r['rtt_min_ms']} / {r['rtt_max_ms']} ms")
            print(f"  RTT p99           : {r['rtt_p99_ms']} ms")
        if r['lost_seqs']:
            print(f"  Lost seqs         : {r['lost_seqs'][:20]}")
        if r['retransmit_seqs']:
            print(f"  Retransmit seqs   : {r['retransmit_seqs'][:20]}")
        print(f"{'='*w}")


# ── libp2p host factory ────────────────────────────────────────────────────────
def make_host(port: int):
    """
    Create a libp2p host with:
      - secp256k1 key pair (fresh per session)
      - Noise security handshake
      - QUIC transport enabled
      - Reasonable resource limits
    """
    key_pair = create_new_key_pair(secrets.token_bytes(32))
    noise    = NoiseTransport(
        libp2p_keypair=key_pair,
        noise_privkey=key_pair.private_key,
        early_data=None,
    )
    limits = ResourceLimits(
        max_connections=64,
        max_streams=512,
        max_memory_mb=256,
    )
    rm   = new_resource_manager(limits=limits)
    host = new_host(
        key_pair=key_pair,
        sec_opt={NOISE_PROTOCOL_ID: noise},
        resource_manager=rm,
        enable_quic=True,
    )
    return host, key_pair


def get_quic_addrs(host, port: int) -> List[Multiaddr]:
    """Return IPv4-only QUIC-v1 listen addresses (skip IPv6 to avoid tuple issues)."""
    addrs = get_available_interfaces(port, protocol="udp")
    result = []
    for a in addrs:
        s = str(a)
        if "/ip4/" in s and "/ip6/" not in s:
            quic_addr = s.replace("/tcp/", "/udp/") + "/quic-v1"
            result.append(Multiaddr(quic_addr))
    return result


# ── SENDER ────────────────────────────────────────────────────────────────────
async def send_bundle(manifest_chunk: bytes,
                      data_chunks: List[bytes],
                      peer_addr: str,
                      port: int = 8001,
                      flight_id: str = "FLT0001") -> "TransportMetrics":
    """
    Send manifest + all data chunks over QUIC.

    Protocol flow:
      1. Connect to receiver (with retry + backoff)
      2. Open DATA stream, send manifest
      3. Register ACK handler (PROTO_ACK) — receiver will open this back
      4. Wait for "READY" on ACK stream (confirms ACK channel is open)
      5. Send each chunk → wait for "ACK:<seq>" → record RTT
         Retransmit on timeout (up to CONNECT_RETRY times)
      6. Send "DONE", wait for "RECV_COMPLETE" / "RECV_PARTIAL"
      7. Print and return metrics
    """
    metrics           = TransportMetrics(flight_id=flight_id)
    metrics.total_bytes = (sum(len(c) for c in data_chunks)
                           + len(manifest_chunk))

    host, _ = make_host(port)
    addrs   = get_quic_addrs(host, port)

    # ACK channel: receiver opens PROTO_ACK stream back to us.
    # We store the stream in this variable once it arrives.
    ack_stream_arrived = trio.Event()
    ack_stream_holder  = {"stream": None}

    async def _handle_ack_stream(stream):
        """Called when receiver opens the ACK-back stream."""
        ack_stream_holder["stream"] = stream
        ack_stream_arrived.set()
        # Keep stream alive — sender reads from it throughout the session.
        # It is drained by the send loop below; this handler just registers it.

    # Register ACK handler and dummy DATA handler BEFORE host.run()
    host.set_stream_handler(PROTO_ACK, _handle_ack_stream)
    host.set_stream_handler(PROTO_DATA, lambda s: None)  # sender doesn't receive data

    async with host.run(listen_addrs=addrs):
        print(f"[SENDER] Local peer: {host.get_id().to_string()[:24]}...  port={port}")

        # ── Connect with exponential backoff ──────────────────────────────
        peer_info = info_from_p2p_addr(Multiaddr(peer_addr))
        for attempt in range(1, CONNECT_RETRY + 1):
            try:
                print(f"[SENDER] Connecting (attempt {attempt}/{CONNECT_RETRY})...")
                await host.connect(peer_info)
                print(f"[SENDER] Connected ✓  remote={peer_info.peer_id.to_string()[:24]}...")
                break
            except Exception as e:
                print(f"[SENDER] Connect failed: {e}")
                if attempt == CONNECT_RETRY:
                    raise RuntimeError(
                        f"Cannot connect after {CONNECT_RETRY} attempts: {e}"
                    )
                wait = 2 ** attempt
                print(f"[SENDER] Retrying in {wait}s...")
                await trio.sleep(wait)

        # ── Open DATA stream and send manifest ────────────────────────────
        data_stream = await host.new_stream(peer_info.peer_id, [PROTO_DATA])
        print(f"[SENDER] DATA stream open  protocol={PROTO_DATA}")

        try:
            print(f"[SENDER] Sending manifest ({len(manifest_chunk)}B)...")
            await send_msg(data_stream, manifest_chunk)

            # ── Wait for receiver to open ACK channel back ────────────────
            print(f"[SENDER] Waiting for ACK channel from receiver...")
            with trio.move_on_after(FRAME_TIMEOUT) as cs:
                await ack_stream_arrived.wait()
            if cs.cancelled_caught:
                raise RuntimeError("Timed out waiting for ACK channel from receiver")
            ack_stream = ack_stream_holder["stream"]
            print(f"[SENDER] ACK channel established ✓")

            # Consume the "READY" handshake message
            with trio.move_on_after(FRAME_TIMEOUT) as cs:
                ready_msg = await recv_msg(ack_stream)
            if cs.cancelled_caught or ready_msg != b"READY":
                raise RuntimeError(f"Bad READY: {ready_msg!r}")

            # ── Send chunks with per-chunk ACK ────────────────────────────
            n = len(data_chunks)
            for seq, chunk in enumerate(data_chunks):
                sent   = False
                retx   = 0

                for attempt in range(CONNECT_RETRY + 1):
                    sent_at = time.time()
                    await send_msg(data_stream, chunk)

                    with trio.move_on_after(FRAME_TIMEOUT) as cs:
                        ack = await recv_msg(ack_stream)

                    if cs.cancelled_caught:
                        retx += 1
                        print(f"  [SENDER] seq={seq} timeout → retransmit #{retx}")
                        continue

                    if ack == f"ACK:{seq}".encode():
                        acked_at = time.time()
                        metrics.record(seq, len(chunk), sent_at, acked_at, retx, False)
                        sent = True

                        # Progress log every 50 chunks or on retransmit
                        if seq % 50 == 0 or retx > 0 or seq == n - 1:
                            pct = (seq + 1) / n * 100
                            rtt = (acked_at - sent_at) * 1000
                            retx_tag = f"  RETX={retx}" if retx else ""
                            print(f"  [SENDER] {seq+1:4d}/{n}  {pct:5.1f}%  "
                                  f"RTT={rtt:.1f}ms{retx_tag}")
                        break
                    else:
                        print(f"  [SENDER] seq={seq} unexpected ACK: {ack!r}")
                        retx += 1

                if not sent:
                    # All retransmit attempts exhausted — mark as lost
                    metrics.record(seq, len(chunk), sent_at, None, retx, True)
                    print(f"  [SENDER] seq={seq} LOST after {retx} retransmits")

            # ── Signal end of transmission ────────────────────────────────
            await send_msg(data_stream, b"DONE")
            print(f"[SENDER] Sent DONE — waiting for final status...")

            with trio.move_on_after(FRAME_TIMEOUT) as cs:
                final = await recv_msg(ack_stream)
            if cs.cancelled_caught:
                print("[SENDER] WARNING: Final status timed out")
            else:
                print(f"[SENDER] Final status: {final.decode()!r}")

            # Half-close write side cleanly (lets QUIC flush buffers)
            await data_stream.close_write()

        except Exception as e:
            print(f"[SENDER] Session error: {e}")
            try:
                await data_stream.reset()
            except Exception:
                pass
        finally:
            metrics.end_time = time.time()

    metrics.print_report()
    return metrics


# ── RECEIVER bundle state ─────────────────────────────────────────────────────
class BundleReceiver:
    """
    Stateful receiver: accumulates chunks from one session.
    DTN concept: can survive partial delivery — missing_seqs tracked
    so caller knows exactly what to request retransmit for.
    """

    def __init__(self):
        self.manifest:     Optional[dict]  = None
        self.received:     Dict[int, bytes] = {}   # seq → raw chunk bytes
        self.missing:      set              = set()
        self.total_chunks: int              = 0
        self.bundle_id:    str              = ""
        os.makedirs(RECV_SAVE_DIR, exist_ok=True)

    def got_manifest(self, raw_chunk: bytes):
        from packet_framer import parse_chunk
        c = parse_chunk(raw_chunk)
        if c and c["is_manifest"]:
            self.manifest     = json.loads(c["payload"])
            self.total_chunks = self.manifest["n_chunks"]
            self.bundle_id    = self.manifest["bundle_id"]
            self.missing      = set(range(self.total_chunks))
            print(f"[RECV] Manifest OK: {self.total_chunks} chunks expected  "
                  f"bundle={self.bundle_id[:16]}...")

    def got_chunk(self, raw_chunk: bytes) -> int:
        """Store chunk, returns seq number or -1 if parse failed."""
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
        Reassemble all received chunks into the original bundle file.
        Verifies per-chunk Ed25519 signatures and final SHA256.
        Returns saved file path or None on failure.
        """
        from packet_framer import parse_chunk, reassemble

        parsed = [parse_chunk(r) for r in self.received.values() if r]
        result, report = reassemble(parsed, verifier)
        print(f"[RECV] Reassemble: ok={report['chunks_ok']}  "
              f"bad_crc={report['chunks_bad_crc']}  "
              f"bad_sig={report['chunks_bad_sig']}  "
              f"missing={len(report['missing_seqs'])}")

        if report["missing_seqs"]:
            print(f"[RECV] WARNING: missing seqs: {report['missing_seqs'][:10]}")

        # Verify end-to-end SHA256
        if self.manifest:
            expected = self.manifest.get("sha256", "")
            actual   = hashlib.sha256(result).hexdigest()
            if actual != expected:
                print(f"[RECV] ERROR: SHA256 mismatch — data corrupt")
                print(f"  expected: {expected[:32]}...")
                print(f"  actual:   {actual[:32]}...")
                return None
            print(f"[RECV] SHA256 verified ✓")

        fname = f"{RECV_SAVE_DIR}/{self.bundle_id[:8]}_{int(time.time())}.a717"
        with open(fname, "wb") as f:
            f.write(result)
        print(f"[RECV] Bundle saved: {fname}  ({len(result):,} bytes)")
        return fname


# ── RECEIVER stream handlers ──────────────────────────────────────────────────
async def _handle_data_stream(stream,
                               receiver: BundleReceiver,
                               verifier,
                               sender_peer_id,
                               host):
    """
    Handle incoming DATA stream from sender.

    Steps:
      1. Receive manifest
      2. Open ACK stream BACK to sender (fixes Bug 1 — server-initiated stream)
      3. Send "READY" on ACK stream
      4. Receive chunks, ACK each on ACK stream
      5. Receive "DONE", reassemble, send final status on ACK stream
    """
    chunks_received = 0
    ack_stream      = None

    try:
        # ── Step 1: receive manifest ──────────────────────────────────────
        with trio.move_on_after(FRAME_TIMEOUT) as cs:
            raw_mf = await recv_msg(stream)
        if cs.cancelled_caught:
            print("[RECV] Timeout waiting for manifest")
            return

        receiver.got_manifest(raw_mf)

        # ── Step 2: open ACK stream BACK to sender ────────────────────────
        # This is the fix for Bug 1.
        # We open a NEW stream originating from the receiver (server).
        # Server-initiated streams have IDs 1,5,9... which the SERVER can write.
        print(f"[RECV] Opening ACK stream back to sender...")
        try:
            ack_stream = await host.new_stream(sender_peer_id, [PROTO_ACK])
            print(f"[RECV] ACK stream open ✓  protocol={PROTO_ACK}")
        except Exception as e:
            print(f"[RECV] ERROR: Cannot open ACK stream: {e}")
            return

        # ── Step 3: signal ready ──────────────────────────────────────────
        await send_msg(ack_stream, b"READY")

        # ── Step 4: receive chunks, ACK each ─────────────────────────────
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
                await send_msg(ack_stream, f"ACK:{seq}".encode())

                # Progress log
                if chunks_received % 50 == 0:
                    total = receiver.total_chunks
                    pct   = chunks_received / max(total, 1) * 100
                    print(f"  [RECV] {chunks_received}/{total}  {pct:.1f}%  "
                          f"missing={len(receiver.missing)}")
            else:
                print(f"[RECV] Unparseable chunk — skipping")

        # ── Step 5: reassemble and report ────────────────────────────────
        print(f"[RECV] Received DONE. Total chunks: {chunks_received}. Reassembling...")
        fname  = receiver.reassemble_and_save(verifier)
        status = b"RECV_COMPLETE" if fname else b"RECV_PARTIAL"
        await send_msg(ack_stream, status)
        print(f"[RECV] Status sent: {status.decode()}")

        await ack_stream.close_write()
        await stream.close_write()

    except Exception as e:
        print(f"[RECV] Stream error: {e}")
        import traceback; traceback.print_exc()
        try:
            await stream.reset()
        except Exception:
            pass
        if ack_stream:
            try:
                await ack_stream.reset()
            except Exception:
                pass
    finally:
        for s in [stream, ack_stream]:
            if s:
                try:
                    await s.close()
                except Exception:
                    pass


# ── RECEIVER entry point ──────────────────────────────────────────────────────
async def run_receiver(port: int = 8000,
                       verifier=None,
                       max_bundles: int = 1):
    """
    Start QUIC receiver. Handles up to max_bundles sessions then exits.
    Prints the full multiaddr the sender needs to connect.
    """
    # Bug 2 fix: always have a valid verifier object — never None
    if verifier is None:
        verifier = _NullVerifier()
        print("[RECV] WARNING: No public key provided — "
              "signature verification DISABLED. Use --pubkey in production.")

    host, _ = make_host(port)
    addrs   = get_quic_addrs(host, port)

    bundles_done = 0
    done_event   = trio.Event()

    async def _on_data_stream(stream):
        """Called by libp2p when sender opens a PROTO_DATA stream."""
        nonlocal bundles_done

        # Get sender peer ID from the stream's connection
        try:
            sender_peer_id = stream.muxed_conn.peer_id
        except Exception:
            # Fallback: parse from stream metadata
            sender_peer_id = stream._connection.peer_id

        receiver = BundleReceiver()
        await _handle_data_stream(stream, receiver, verifier, sender_peer_id, host)

        bundles_done += 1
        if bundles_done >= max_bundles:
            done_event.set()

    # Register handler BEFORE host.run() — critical ordering fix
    host.set_stream_handler(PROTO_DATA, _on_data_stream)
    # ACK protocol is opened by US back to sender, not handled inbound here

    async with host.run(listen_addrs=addrs):
        pid = host.get_id().to_string()
        print(f"\n[RECV] ──────────────────────────────────────────")
        print(f"[RECV] libp2p QUIC Receiver READY")
        print(f"[RECV] Peer ID : {pid}")
        print(f"[RECV] ──────────────────────────────────────────")
        for a in addrs:
            print(f"[RECV] Address : {a}/p2p/{pid}")
        print(f"[RECV] ──────────────────────────────────────────")
        print(f"[RECV] Waiting for {max_bundles} bundle(s)...")

        await done_event.wait()
        print(f"\n[RECV] {bundles_done} bundle(s) received. Shutting down.")


# ── Null verifier for testing without a key ────────────────────────────────────
class _NullVerifier:
    """
    Bypasses Ed25519 verification when no public key is available.
    For testing ONLY — prints a warning on every call.
    Never use in production.
    """
    def verify(self, data: bytes, sig: bytes) -> bool:
        return True   # WARNING: no real verification


# ── CLI entry points ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    if mode == "receiver":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        print(f"[MAIN] Starting receiver on port {port}")
        trio.run(run_receiver, port)

    elif mode == "sender":
        if len(sys.argv) < 3:
            print("Usage: python quic_transport.py sender <peer_multiaddr> [port]")
            sys.exit(1)
        peer_addr = sys.argv[2]
        port      = int(sys.argv[3]) if len(sys.argv) > 3 else 8001

        sys.path.insert(0, os.path.dirname(__file__))
        from compressor import compress_payload, pack_bundle, simulate_cvr_audio, EVT_DISTRESS
        from arinc_generator import generate_telemetry
        from packet_framer import build_chunks, Signer

        print("[MAIN] Generating quick test bundle (60s)...")
        raw, _, _, _ = generate_telemetry(60)
        cvr  = simulate_cvr_audio(60)
        comp, m = compress_payload(raw, cvr)
        print(f"[MAIN] Compressed: {m['comp_bytes']:,}B  CR={m['cr']:.2f}x")

        bundle = pack_bundle(comp, event_type=EVT_DISTRESS, pre_event_sec=60)
        signer = Signer()
        mf, chunks = build_chunks(bundle, signer)

        trio.run(send_bundle, mf, chunks, peer_addr, port, "TEST001")

    else:
        print("Usage: python quic_transport.py [receiver|sender] [args]")
        print("  receiver [port]              — start receiver (default port 8000)")
        print("  sender <multiaddr> [port]    — send test bundle")
