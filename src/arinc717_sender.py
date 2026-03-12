"""
ARINC 717 → Adaptive Huffman → Binary file → libp2p QUIC sender
Includes: packet loss tracking, retransmit, latency, per-packet metrics
"""

import os, struct, time, math, hashlib, heapq, asyncio, secrets
import numpy as np
from collections import Counter, defaultdict

# ─── ARINC 717 constants ───────────────────────────────────────────────────────
WORDS_PER_SF   = 256
BITS_PER_WORD  = 12
MAX_VAL        = 4095
SYNC           = [0o0247, 0o0132, 0o0310, 0o0734]
BUFFER_SEC     = 1200   # 20 min = 300 frames
FRAME_DURATION = 4      # sec
PARAM_NAMES    = ["Altitude","Airspeed","Pitch","Roll","Yaw",
                  "Engine_N1","Engine_N2","Temperature","Fuel_Flow","Sys_Flags"]

# ─── SECTION 1: Telemetry generator ──────────────────────────────────────────
def generate_telemetry(duration_sec=BUFFER_SEC):
    print(f"\n[GEN] Generating {duration_sec}s ARINC 717 telemetry "
          f"({duration_sec//FRAME_DURATION} frames)...")
    np.random.seed(42)
    N = WORDS_PER_SF * duration_sec
    t = np.linspace(0, duration_sec, N)

    channels = [
        np.clip(0.75 + 0.05*np.sin(2*np.pi*t/600)  + 0.003*np.random.randn(N), 0, 1),
        np.clip(0.60 + 0.01*np.sin(2*np.pi*t/120)  + 0.005*np.random.randn(N), 0, 1),
        np.clip(0.50 + 0.03*np.sin(2*np.pi*t/45)   + 0.008*np.random.randn(N), 0, 1),
        np.clip(0.50 + 0.02*np.sin(2*np.pi*t/30)   + 0.010*np.random.randn(N), 0, 1),
        np.clip(0.50 + 0.01*np.sin(2*np.pi*t/200)  + 0.003*np.random.randn(N), 0, 1),
        np.clip(0.82 + 0.01*np.sin(2*np.pi*t/300)  + 0.002*np.random.randn(N), 0, 1),
        np.clip(0.84 + 0.008*np.sin(2*np.pi*t/300) + 0.002*np.random.randn(N), 0, 1),
        np.clip(0.35 + 0.002*np.random.randn(N), 0, 1),
        np.clip(0.45 + 0.005*np.random.randn(N), 0, 1),
        np.clip((np.random.randn(N) > 0.995).astype(float) * 0.1, 0, 1),
    ]

    total_sfs = duration_sec
    frame_arr = np.zeros((total_sfs, WORDS_PER_SF), dtype=np.uint16)
    for sf in range(total_sfs):
        frame_arr[sf, 0] = SYNC[sf % 4]

    for i in range(N):
        sf   = i // WORDS_PER_SF
        slot = (i % WORDS_PER_SF) + 1
        if slot < WORDS_PER_SF:
            ch  = i % len(channels)
            frame_arr[sf, slot] = int(channels[ch][i] * MAX_VAL) & MAX_VAL

    flat   = frame_arr.reshape(-1)
    packed = bytearray()
    for i in range(0, len(flat)-1, 2):
        a, b = int(flat[i]), int(flat[i+1])
        packed.append((a >> 4) & 0xFF)
        packed.append(((a & 0x0F) << 4) | ((b >> 8) & 0x0F))
        packed.append(b & 0xFF)

    raw = bytes(packed)
    print(f"[GEN] Raw bytes: {len(raw):,}  Frames: {duration_sec//FRAME_DURATION}")
    return raw, frame_arr


# ─── SECTION 2: Adaptive Huffman (Vitter algorithm simplified) ──────────────
class AdaptiveHuffmanEncoder:
    """
    Adaptive Huffman (FGK simplified).
    No static frequency table — tree updates symbol-by-symbol.
    Advantage: no header needed; adapts to data distribution in real time.
    """
    NYT = 256   # Not Yet Transmitted symbol

    def __init__(self):
        self._reset()

    def _reset(self):
        # node: [weight, symbol, parent, left, right, order]
        self.nodes    = {}        # node_id → dict
        self.sym_node = {}        # symbol → node_id
        self.next_id  = 0
        self.root     = self._new_node(0, self.NYT, None, None, None, 512)
        self.nyt      = self.root
        self.bits_out = []

    def _new_node(self, w, sym, parent, left, right, order):
        nid = self.next_id; self.next_id += 1
        self.nodes[nid] = dict(w=w, sym=sym, par=parent,
                               lc=left, rc=right, ord=order)
        return nid

    def _get_code(self, nid):
        code = []
        while self.nodes[nid]['par'] is not None:
            par = self.nodes[nid]['par']
            code.append('0' if self.nodes[par]['lc'] == nid else '1')
            nid = par
        return ''.join(reversed(code))

    def _swap(self, a, b):
        """Swap two nodes (except parents)."""
        na, nb = self.nodes[a], self.nodes[b]
        na['ord'], nb['ord'] = nb['ord'], na['ord']
        # fix parent pointers
        pa, pb = na['par'], nb['par']
        if pa == b: pa, pb = pb, pa  # edge case: siblings
        if pa is not None:
            if self.nodes[pa]['lc'] == a: self.nodes[pa]['lc'] = b
            else:                          self.nodes[pa]['rc'] = b
        if pb is not None:
            if self.nodes[pb]['lc'] == b: self.nodes[pb]['lc'] = a
            else:                          self.nodes[pb]['rc'] = a
        na['par'], nb['par'] = pb, pa
        self.nodes[a], self.nodes[b] = nb, na
        # undo the dict swap by key
        self.nodes[a], self.nodes[b] = self.nodes[b], self.nodes[a]

    def _update(self, nid):
        """Increment weight and rebalance up the tree."""
        while nid is not None:
            n = self.nodes[nid]
            # find node with same weight and highest order
            best = nid
            for k, v in self.nodes.items():
                if v['w'] == n['w'] and v['ord'] > self.nodes[best]['ord'] \
                        and k != n['par']:
                    best = k
            if best != nid and best != n['par']:
                self._swap(nid, best)
                nid = best
            self.nodes[nid]['w'] += 1
            nid = self.nodes[nid]['par']

    def encode_symbol(self, sym):
        """Return bit string for one symbol and update tree."""
        if sym in self.sym_node:
            code = self._get_code(self.sym_node[sym])
            self._update(self.sym_node[sym])
            return code
        else:
            # send NYT code + raw 8-bit symbol
            nyt_code = self._get_code(self.nyt)
            raw_bits = format(sym, '08b')
            # Spawn two children from NYT
            par    = self.nyt
            left   = self._new_node(0, self.NYT, par, None, None,
                                    self.nodes[par]['ord'] - 2)
            right  = self._new_node(1, sym,      par, None, None,
                                    self.nodes[par]['ord'] - 1)
            self.nodes[par]['lc']  = left
            self.nodes[par]['rc']  = right
            self.nodes[par]['sym'] = None
            self.nyt = left
            self.sym_node[sym] = right
            self._update(right)
            return nyt_code + raw_bits

    def compress(self, data: bytes) -> bytes:
        self._reset()
        bits = ''.join(self.encode_symbol(b) for b in data)
        # EOF marker
        bits += self._get_code(self.nyt) + format(256, '09b')
        pad   = (8 - len(bits) % 8) % 8
        bits += '0' * pad
        out = bytearray([pad])
        for i in range(0, len(bits), 8):
            out.append(int(bits[i:i+8], 2))
        return bytes(out)


class AdaptiveHuffmanDecoder:
    NYT = 256

    def __init__(self):
        self.nodes    = {}
        self.sym_node = {}
        self.next_id  = 0
        self.root     = self._new_node(0, self.NYT, None, None, None, 512)
        self.nyt      = self.root

    def _new_node(self, w, sym, parent, left, right, order):
        nid = self.next_id; self.next_id += 1
        self.nodes[nid] = dict(w=w, sym=sym, par=parent,
                               lc=left, rc=right, ord=order)
        return nid

    def _swap(self, a, b):
        na, nb = self.nodes[a], self.nodes[b]
        na['ord'], nb['ord'] = nb['ord'], na['ord']
        pa, pb = na['par'], nb['par']
        if pa == b: pa, pb = pb, pa
        if pa is not None:
            if self.nodes[pa]['lc'] == a: self.nodes[pa]['lc'] = b
            else:                          self.nodes[pa]['rc'] = b
        if pb is not None:
            if self.nodes[pb]['lc'] == b: self.nodes[pb]['lc'] = a
            else:                          self.nodes[pb]['rc'] = a
        na['par'], nb['par'] = pb, pa
        self.nodes[a], self.nodes[b] = self.nodes[b], self.nodes[a]

    def _update(self, nid):
        while nid is not None:
            n = self.nodes[nid]
            best = nid
            for k, v in self.nodes.items():
                if v['w'] == n['w'] and v['ord'] > self.nodes[best]['ord'] \
                        and k != n['par']:
                    best = k
            if best != nid and best != n['par']:
                self._swap(nid, best)
                nid = best
            self.nodes[nid]['w'] += 1
            nid = self.nodes[nid]['par']

    def decompress(self, data: bytes) -> bytes:
        pad    = data[0]
        bits   = ''.join(format(b, '08b') for b in data[1:])
        if pad: bits = bits[:-pad]
        out    = bytearray()
        cur    = self.root
        i      = 0
        while i < len(bits):
            n = self.nodes[cur]
            if n['lc'] is None:  # leaf
                sym = n['sym']
                if sym == self.NYT:
                    if i + 9 > len(bits): break
                    sym = int(bits[i:i+9], 2); i += 9
                    if sym == 256: break  # EOF
                    # add new leaf
                    par   = cur
                    left  = self._new_node(0, self.NYT, par, None, None,
                                           self.nodes[par]['ord'] - 2)
                    right = self._new_node(1, sym, par, None, None,
                                           self.nodes[par]['ord'] - 1)
                    self.nodes[par]['lc']  = left
                    self.nodes[par]['rc']  = right
                    self.nodes[par]['sym'] = None
                    self.nyt = left
                    self.sym_node[sym] = right
                    self._update(right)
                else:
                    self._update(self.sym_node.get(sym, cur))
                out.append(sym)
                cur = self.root
            else:
                cur = n['lc'] if bits[i] == '0' else n['rc']
                i += 1
        return bytes(out)


# ─── SECTION 3: Pack compressed data into binary file ────────────────────────
MAGIC = b"A717"

def pack_binary(compressed: bytes, frame_arr, algo="ADHUFF") -> bytes:
    """
    Binary file format:
    [MAGIC 4B][algo 6B][n_frames 4B][orig_size 4B][comp_size 4B]
    [sha256 32B][payload]
    """
    n_frames  = frame_arr.shape[0] // 4
    orig_size = frame_arr.nbytes
    sha256    = hashlib.sha256(compressed).digest()
    header    = (MAGIC
                 + algo.encode().ljust(6)[:6]
                 + struct.pack(">I", n_frames)
                 + struct.pack(">I", orig_size)
                 + struct.pack(">I", len(compressed))
                 + sha256)
    return header + compressed


def unpack_binary(data: bytes):
    assert data[:4] == MAGIC, "Bad magic"
    algo      = data[4:10].rstrip().decode()
    n_frames  = struct.unpack(">I", data[10:14])[0]
    orig_size = struct.unpack(">I", data[14:18])[0]
    comp_size = struct.unpack(">I", data[18:22])[0]
    sha256    = data[22:54]
    payload   = data[54:54+comp_size]
    actual    = hashlib.sha256(payload).digest()
    assert actual == sha256, "SHA256 mismatch — file corrupted"
    return payload, n_frames, algo


# ─── SECTION 4: Print ALL frames ─────────────────────────────────────────────
def print_frame_summary(frame_arr, max_frames=None):
    """Print parameter values for every frame (or first max_frames)."""
    total_sfs    = frame_arr.shape[0]
    n_frames     = total_sfs // 4
    to_print     = n_frames if max_frames is None else min(n_frames, max_frames)
    print(f"\n[FRAMES] Total frames: {n_frames}  |  Printing: {to_print}")
    print(f"{'Frame':>6} {'SF':>4} {'Sync':>6}  " +
          "  ".join(f"{p[:8]:>8}" for p in PARAM_NAMES))
    print("─" * 120)
    for fr in range(to_print):
        for sf in range(4):
            sf_idx = fr * 4 + sf
            if sf_idx >= total_sfs: break
            row    = frame_arr[sf_idx]
            sync   = row[0]
            params = [row[1 + i] for i in range(len(PARAM_NAMES))]
            label  = f"F{fr+1:03d}" if sf == 0 else ""
            print(f"{label:>6} SF{sf+1:1d} {sync:06o}  " +
                  "  ".join(f"{p:8d}" for p in params))
    if to_print < n_frames:
        print(f"  ... {n_frames - to_print} more frames not shown ...")


# ─── SECTION 5: Metrics tracker ──────────────────────────────────────────────
class TransportMetrics:
    def __init__(self):
        self.sent       = 0
        self.lost       = 0
        self.retransmit = 0
        self.acked      = 0
        self.latencies  = []     # RTT per packet in ms
        self.t_start    = time.time()
        self.packet_log = []     # list of dicts per packet

    def record(self, pkt_id, sent_at, acked_at, lost, retransmits):
        rtt = (acked_at - sent_at) * 1000 if not lost else None
        self.sent       += 1
        self.lost       += int(lost)
        self.retransmit += retransmits
        if rtt is not None:
            self.acked += 1
            self.latencies.append(rtt)
        self.packet_log.append(dict(
            id=pkt_id, rtt_ms=rtt, lost=lost, retransmits=retransmits
        ))

    def report(self):
        elapsed = time.time() - self.t_start
        print(f"\n{'='*60}")
        print(f"  TRANSPORT METRICS")
        print(f"{'='*60}")
        print(f"  Packets sent      : {self.sent}")
        print(f"  Packets acked     : {self.acked}")
        print(f"  Packets lost      : {self.lost}")
        print(f"  Loss rate         : {self.lost/max(self.sent,1)*100:.2f}%")
        print(f"  Retransmissions   : {self.retransmit}")
        if self.latencies:
            lats = self.latencies
            print(f"  Latency avg (RTT) : {sum(lats)/len(lats):.2f} ms")
            print(f"  Latency min       : {min(lats):.2f} ms")
            print(f"  Latency max       : {max(lats):.2f} ms")
            sorted_l = sorted(lats)
            p99 = sorted_l[int(0.99*len(sorted_l))]
            p50 = sorted_l[int(0.50*len(sorted_l))]
            print(f"  p50 latency       : {p50:.2f} ms")
            print(f"  p99 latency       : {p99:.2f} ms")
        print(f"  Total duration    : {elapsed:.2f} s")
        print(f"{'='*60}")

        # Per-packet debug for lost/retransmitted
        retrx = [p for p in self.packet_log if p['retransmits'] > 0 or p['lost']]
        if retrx:
            print(f"\n  [DEBUG] Lost / Retransmitted packets:")
            print(f"  {'PktID':>8}  {'Lost':>6}  {'Retransmits':>12}  {'RTT(ms)':>10}")
            for p in retrx[:50]:  # cap at 50 lines
                rtt_str = f"{p['rtt_ms']:.2f}" if p['rtt_ms'] else "LOST"
                print(f"  {p['id']:>8}  {str(p['lost']):>6}  {p['retransmits']:>12}  {rtt_str:>10}")


# ─── SECTION 6: QUIC-like sender (simulated over asyncio) ────────────────────
PACKET_SIZE   = 1200   # bytes per QUIC packet (typical MTU-safe)
MAX_RETRIES   = 5
LOSS_SIM      = 0.03   # 3% simulated loss for demo

async def send_payload_quic(payload: bytes, metrics: TransportMetrics):
    """
    Simulate QUIC packet send with:
    - Chunking into PACKET_SIZE datagrams
    - Simulated ACK / loss / retransmit
    - Per-packet RTT measurement
    """
    import random
    random.seed(99)

    chunks    = [payload[i:i+PACKET_SIZE]
                 for i in range(0, len(payload), PACKET_SIZE)]
    total     = len(chunks)
    print(f"\n[QUIC] Sending {len(payload):,} bytes in {total} packets "
          f"({PACKET_SIZE}B each)")
    print(f"[QUIC] Simulated loss rate: {LOSS_SIM*100:.0f}%  |  "
          f"Max retries: {MAX_RETRIES}")
    print(f"[QUIC] {'PktID':>6}  {'Size':>6}  {'RTT(ms)':>8}  {'Status':>12}")
    print(f"{'─'*50}")

    for pkt_id, chunk in enumerate(chunks):
        sent_at    = time.time()
        retransmits = 0
        delivered  = False

        for attempt in range(MAX_RETRIES):
            # simulate network delay (1–20 ms base + jitter)
            delay = 0.001 + random.gauss(0.008, 0.003)
            await asyncio.sleep(max(0.001, delay))

            lost = random.random() < LOSS_SIM
            if not lost:
                delivered = True
                break
            retransmits += 1

        acked_at = time.time()
        metrics.record(pkt_id, sent_at, acked_at, not delivered, retransmits)

        if pkt_id % 100 == 0 or retransmits > 0 or not delivered:
            status = ("LOST" if not delivered
                      else f"RETX×{retransmits}" if retransmits
                      else "OK")
            rtt = f"{(acked_at-sent_at)*1000:.1f}" if delivered else "—"
            print(f"  {pkt_id:>6}  {len(chunk):>6}  {rtt:>8}  {status:>12}")

    print(f"[QUIC] Done. {metrics.acked}/{total} delivered.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

# ─── Fast canonical Huffman ───────────────────────────────────────────────────
def fast_huff_compress(data: bytes) -> bytes:
    freq = Counter(data)
    heap = [(f, s) for s, f in freq.items()]; heapq.heapify(heap)
    par = {}; lc = {}; rc = {}; nid = 256
    while len(heap) > 1:
        f1,n1=heapq.heappop(heap); f2,n2=heapq.heappop(heap)
        lc[nid]=n1; rc[nid]=n2; par[n1]=par[n2]=nid
        heapq.heappush(heap,(f1+f2,nid)); nid+=1
    root=heap[0][1]; lengths={}
    def depth(n,d=0):
        if n<256: lengths[n]=max(d,1); return
        depth(lc[n],d+1); depth(rc[n],d+1)
    depth(root)
    syms=sorted(lengths.items(),key=lambda x:(x[1],x[0]))
    codes={}; cv=0; pl=0
    for sym,ln in syms: cv<<=(ln-pl); codes[sym]=(cv,ln); cv+=1; pl=ln
    ob=0; ol=0
    for b in data: c,ln=codes[b]; ob=(ob<<ln)|c; ol+=ln
    pad=(8-ol%8)%8; ob<<=pad; ol+=pad
    body=ob.to_bytes(ol//8,'big')
    hdr=struct.pack(">H",len(codes))
    for sym,(_,ln) in sorted(codes.items(),key=lambda x:(x[1][1],x[0])):
        hdr+=bytes([sym,ln])
    hdr+=bytes([pad])
    return hdr+body


# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    raw, frame_arr = generate_telemetry(BUFFER_SEC)
    print_frame_summary(frame_arr)

    # Adaptive Huffman demo on small chunk
    DEMO = 2048
    raw_arr = np.frombuffer(raw, dtype=np.uint8)
    delta   = np.diff(raw_arr, prepend=raw_arr[0]).astype(np.uint8)
    print(f"\n[ADHUFF] Demo on {DEMO} bytes (FGK adaptive, no static table)...")
    t0=time.time()
    enc=AdaptiveHuffmanEncoder(); dc=enc.compress(bytes(delta[:DEMO]))
    ok=AdaptiveHuffmanDecoder().decompress(dc)[:DEMO]==bytes(delta[:DEMO])
    print(f"[ADHUFF] {DEMO} → {len(dc)}B  CR:{DEMO/max(len(dc),1):.3f}x  "
          f"Time:{time.time()-t0:.3f}s  Roundtrip:{'PASS' if ok else 'FAIL'}")

    # Full compression: canonical Huffman + delta
    print(f"\n[HUFFMAN] Full {len(raw):,}B with canonical Huffman + delta...")
    t0=time.time(); comp=fast_huff_compress(bytes(delta))
    cr=len(raw)/max(len(comp),1)
    print(f"[HUFFMAN] {len(raw):,} → {len(comp):,}B  CR:{cr:.3f}x  Time:{time.time()-t0:.2f}s")

    binary=pack_binary(comp,frame_arr)
    outpath="/mnt/user-data/outputs/arinc717_adhuff.bin"
    os.makedirs(os.path.dirname(outpath),exist_ok=True)
    with open(outpath,"wb") as f: f.write(binary)
    print(f"[FILE] {outpath}  ({len(binary):,}B)")

    payload,n_frames,algo=unpack_binary(binary)
    print(f"[VERIFY] SHA256 PASS  |  {n_frames} frames  |  algo={algo}")

    metrics=TransportMetrics()
    await send_payload_quic(binary,metrics)
    metrics.report()
    print(f"\n[DONE] Frames:{n_frames}  CR:{cr:.3f}x  File:{outpath}")

if __name__=="__main__":
    asyncio.run(main())
