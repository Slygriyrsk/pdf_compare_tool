"""
MODULE 3 — packet_framer.py
============================
Splits a bundle into transmission chunks, signs each chunk with Ed25519,
and builds a manifest. Implements DTN (Delay/Disruption-Tolerant Networking)
concepts: store-and-forward, custody transfer, per-chunk IDs and ACKs.

Why chunking instead of streaming:
  - 7 kbps link may drop for seconds (ionospheric fade on HF)
  - QUIC handles in-order delivery but we need resume-from-lost-chunk
  - Each chunk is self-contained: ID + sequence + checksum + signature
  - Receiver acknowledges chunks; sender only retransmits missing ones
  - This is the DTN "bundle" concept applied to our use case

Chunk wire format (all big-endian):
  [CHUNK_MAGIC 4B "C717"]
  [bundle_id 16B]
  [chunk_seq 4B uint32]   — 0-based index
  [total_chunks 4B uint32]
  [chunk_len 4B uint32]   — payload bytes in this chunk
  [crc32 4B]              — CRC of chunk payload only
  [sig_len 2B uint16]     — Ed25519 signature length (64 bytes always)
  [signature 64B]         — Ed25519 over (bundle_id + seq + payload)
  [payload ...]

Manifest (sent first as chunk 0xFFFFFFFF):
  JSON with bundle metadata, total chunks, per-chunk expected CRC list.
  Receiver uses manifest to know what to expect and detect missing chunks.
"""

import os
import json
import struct
import time
import hashlib
import uuid
import zlib
from typing import List, Tuple, Optional

import nacl.signing
import nacl.encoding
import nacl.exceptions

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_MAGIC   = b"C717"
CHUNK_HDR_FMT = ">4s 16s I I I I H"      # up to sig; then 64B sig then payload
CHUNK_HDR_SIZE = struct.calcsize(CHUNK_HDR_FMT) + 64   # 4+16+4+4+4+4+2+64 = 102 B
MANIFEST_SEQ  = 0xFFFFFFFF

# Chunk size for QUIC transport (no bandwidth constraint).
# 32 KB keeps each chunk well within a single QUIC stream window,
# large enough to minimise per-chunk overhead while staying retransmittable.
DEFAULT_CHUNK = 32_768


class Signer:
    """
    Ed25519 digital signing.
    Why Ed25519: fast (< 1ms/sign), small sig (64 bytes), secure.
    In real deployment: airplane has private key, ground has public key.
    """

    def __init__(self, private_key_bytes: Optional[bytes] = None):
        if private_key_bytes:
            self.sk = nacl.signing.SigningKey(private_key_bytes)
        else:
            self.sk = nacl.signing.SigningKey.generate()
        self.vk = self.sk.verify_key
        print(f"[SIGNER] Public key: {self.vk.encode(nacl.encoding.HexEncoder).decode()[:32]}...")

    @property
    def private_key_bytes(self) -> bytes:
        return bytes(self.sk)

    @property
    def public_key_bytes(self) -> bytes:
        return bytes(self.vk)

    def sign(self, data: bytes) -> bytes:
        """Returns 64-byte Ed25519 signature."""
        return self.sk.sign(data).signature

    def verify(self, data: bytes, signature: bytes) -> bool:
        """True if signature is valid for data under this key."""
        try:
            self.vk.verify(data, signature)
            return True
        except nacl.exceptions.BadSignatureError:
            return False


class Verifier:
    """Verifies signatures using a known public key (ground station side)."""

    def __init__(self, public_key_bytes: bytes):
        self.vk = nacl.signing.VerifyKey(public_key_bytes)

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            self.vk.verify(data, signature)
            return True
        except nacl.exceptions.BadSignatureError:
            return False


def _signed_data(bundle_id: bytes, seq: int, payload: bytes) -> bytes:
    """
    Canonical byte string that is signed / verified.
    Includes bundle_id + seq so attacker cannot replay a chunk from another bundle.
    """
    return bundle_id + struct.pack(">I", seq) + payload


def build_chunks(bundle_data: bytes,
                 signer: Signer,
                 chunk_size: int = DEFAULT_CHUNK,
                 flight_id: str = "FLT0001",
                 event_type: int = 0) -> Tuple[bytes, List[bytes]]:
    """
    Split bundle_data into signed chunks.

    Returns:
      manifest_chunk : the manifest packet (send first)
      data_chunks    : list of data chunk packets (send in order, or any order
                       since each is self-contained)
    """
    bundle_id = uuid.uuid4().bytes          # 16-byte random ID for this TX session
    total     = len(bundle_data)
    n_chunks  = (total + chunk_size - 1) // chunk_size
    crc_list  = []
    data_chunks = []

    print(f"\n[FRAMER] Chunking {total:,} bytes into {n_chunks} chunks "
          f"of {chunk_size}B  bundle_id={bundle_id.hex()[:16]}...")

    for seq in range(n_chunks):
        start    = seq * chunk_size
        payload  = bundle_data[start:start + chunk_size]
        crc      = zlib.crc32(payload) & 0xFFFFFFFF
        crc_list.append(crc)

        # Sign: bundle_id + seq number + payload
        signed_bytes = _signed_data(bundle_id, seq, payload)
        signature    = signer.sign(signed_bytes)   # 64 bytes

        # Nanosecond UTC timestamp at chunk build time (for latency tracking)
        ts_ns = time.time_ns()

        header = struct.pack(
            CHUNK_HDR_FMT,
            CHUNK_MAGIC,
            bundle_id,
            seq,
            n_chunks,
            len(payload),
            crc,
            64,                  # sig_len always 64 for Ed25519
        )
        # Wire: header(38B) + sig(64B) + ts_ns(8B) + payload
        data_chunks.append(header + signature + struct.pack(">Q", ts_ns) + payload)

    # ── Manifest chunk ────────────────────────────────────────────────────────
    manifest_body = json.dumps({
        "bundle_id":    bundle_id.hex(),
        "flight_id":    flight_id,
        "event_type":   event_type,
        "timestamp":    int(time.time()),
        "timestamp_ns":  time.time_ns(),
        "total_bytes":  total,
        "n_chunks":     n_chunks,
        "chunk_size":   chunk_size,
        "crc_list":     crc_list,
        "pubkey_hex":   signer.public_key_bytes.hex(),
        "sha256":       hashlib.sha256(bundle_data).hexdigest(),
    }).encode()

    mf_crc  = zlib.crc32(manifest_body) & 0xFFFFFFFF
    mf_sig  = signer.sign(_signed_data(bundle_id, MANIFEST_SEQ, manifest_body))
    mf_hdr  = struct.pack(
        CHUNK_HDR_FMT,
        CHUNK_MAGIC, bundle_id, MANIFEST_SEQ, n_chunks,
        len(manifest_body), mf_crc, 64,
    )
    manifest_chunk = mf_hdr + mf_sig + manifest_body

    chunk_sizes = [len(c) for c in data_chunks]
    total_wire  = sum(chunk_sizes) + len(manifest_chunk)
    print(f"[FRAMER] Chunk header overhead: {CHUNK_HDR_SIZE}B per chunk")
    print(f"[FRAMER] Manifest chunk: {len(manifest_chunk)}B")
    print(f"[FRAMER] Total wire bytes: {total_wire:,}  "
          f"overhead: {(total_wire-total)/total*100:.1f}%")

    return manifest_chunk, data_chunks


def parse_chunk(raw: bytes) -> dict:
    """
    Parse a raw chunk packet.
    Returns dict with all header fields + payload.
    Returns None if magic is wrong.
    """
    if len(raw) < CHUNK_HDR_SIZE:
        return None
    hdr_size_no_sig = struct.calcsize(CHUNK_HDR_FMT)
    fields = struct.unpack(CHUNK_HDR_FMT, raw[:hdr_size_no_sig])
    (magic, bundle_id, seq, total_chunks,
     chunk_len, crc, sig_len) = fields

    if magic != CHUNK_MAGIC:
        return None

    off       = hdr_size_no_sig
    signature = raw[off:off + sig_len]; off += sig_len
    # 8-byte nanosecond timestamp (added in v2)
    ts_ns     = struct.unpack_from(">Q", raw, off)[0] if len(raw) >= off + 8 else 0
    off      += 8
    payload   = raw[off:off + chunk_len]
    # Human-readable packet ID
    packet_id = f"{bundle_id.hex()[:8]}_{seq:06d}"

    # Verify CRC
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    crc_ok     = actual_crc == crc

    return {
        "bundle_id":    bundle_id,
        "seq":          seq,
        "total_chunks": total_chunks,
        "chunk_len":    chunk_len,
        "crc":          crc,
        "crc_ok":       crc_ok,
        "ts_sent_ns":   ts_ns,
        "packet_id":    packet_id,
        "sig_len":      sig_len,
        "signature":    signature,
        "payload":      payload,
        "is_manifest":  seq == MANIFEST_SEQ,
    }


def reassemble(chunks: List[dict], verifier: Verifier) -> Tuple[bytes, dict]:
    """
    Reassemble ordered chunks into the original bundle bytes.
    Verifies signatures and CRC on each chunk.

    Returns (bundle_bytes, report_dict).
    report_dict has per-chunk status for debugging.
    """
    # Sort by seq, excluding manifest
    data_chunks = sorted(
        [c for c in chunks if not c["is_manifest"]],
        key=lambda x: x["seq"]
    )

    report = {
        "total_expected": data_chunks[0]["total_chunks"] if data_chunks else 0,
        "total_received": len(data_chunks),
        "chunks_ok":      0,
        "chunks_bad_crc": 0,
        "chunks_bad_sig": 0,
        "missing_seqs":   [],
    }

    received_seqs = {c["seq"] for c in data_chunks}
    expected_seqs = set(range(report["total_expected"]))
    report["missing_seqs"] = sorted(expected_seqs - received_seqs)

    buf = bytearray()
    for c in data_chunks:
        sig_ok  = verifier.verify(
            _signed_data(c["bundle_id"], c["seq"], c["payload"]),
            c["signature"]
        )
        if not c["crc_ok"]:
            report["chunks_bad_crc"] += 1
            print(f"  [REASSEMBLE] WARNING: bad CRC chunk seq={c['seq']}")
        if not sig_ok:
            report["chunks_bad_sig"] += 1
            print(f"  [REASSEMBLE] WARNING: bad SIG chunk seq={c['seq']}")
        if c["crc_ok"] and sig_ok:
            report["chunks_ok"] += 1
        buf.extend(c["payload"])

    return bytes(buf), report


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from compressor import compress_payload, pack_bundle, simulate_cvr_audio
    from arinc_generator import generate_telemetry, BUFFER_SEC

    print("\n=== Packet Framer Test ===")
    raw, frame_arr, slot_map, _ = generate_telemetry(120)   # 2 min for quick test
    cvr = simulate_cvr_audio(120)
    comp, metrics = compress_payload(raw, cvr)
    bundle = pack_bundle(comp, flight_id="FLT0001", event_type=1, pre_event_sec=120)
    print(f"Bundle size: {len(bundle):,} bytes")

    signer   = Signer()
    verifier = Verifier(signer.public_key_bytes)

    mf_chunk, data_chunks = build_chunks(bundle, signer, chunk_size=DEFAULT_CHUNK)

    # Simulate receiving all chunks
    all_raw = [mf_chunk] + data_chunks
    parsed  = [parse_chunk(c) for c in all_raw]

    # Verify manifest
    mf = next(c for c in parsed if c["is_manifest"])
    mf_info = json.loads(mf["payload"])
    print(f"Manifest: n_chunks={mf_info['n_chunks']}  "
          f"sha256={mf_info['sha256'][:16]}...")

    # Reassemble
    data_only = [c for c in parsed if not c["is_manifest"]]
    result, report = reassemble(data_only, verifier)
    print(f"Reassemble: {report}")
    print(f"Bytes match: {result == bundle}")
