# pack_payload_demo.py
# Simulate signals, extract keypoints, detect flagged cycles,
# pack flagged events into 32-bit words with quantization,
# measure bytes & time vs raw send, compute recon MSEs, plot results.
import numpy as np
import time
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

np.random.seed(0)

# ------- params -------
fs = 50            # sampling Hz
T = 30             # seconds
N = int(fs*T)
t = np.arange(N)/fs
names = ['roll','pitch','yaw','roll_rate','pitch_rate','yaw_rate']
freqs = np.array([0.5,0.7,1.1,0.6,0.9,1.3])
amps  = np.array([1.0,1.2,0.9,1.1,0.95,0.8])

# ------- generate signals -------
air_clean = np.zeros((6,N))
for ch in range(6):
    air_clean[ch] = amps[ch]*np.sin(2*np.pi*freqs[ch]*t)
gx_clean = air_clean.copy()

# noise
sig_air = 0.02; sig_gx = 0.008
air = air_clean + sig_air * np.random.randn(6,N)
gx  = gx_clean + sig_gx  * np.random.randn(6,N)

# ------- inject anomalies (step offsets) -------
anoms = [
    (0, 10.0, 0.6, 'offset', 1.5),
    (1, 15.0, 0.8, 'offset', -1.2),
    (2, 20.0, 1.0, 'offset', 2.0),
    (3, 8.5,  0.4, 'offset', 0.9),
    (4, 12.0, 0.7, 'offset', -1.6),
    (5, 22.0, 1.2, 'offset', 1.2),
]
air2 = air.copy()
for ch,s0,L,typ,val in anoms:
    si = int(s0*fs); ei = min(N,int((s0+L)*fs))
    if typ=='offset':
        air2[ch, si:ei] += val

res_full = air2 - gx

# ------- keypoint extraction function -------
from scipy.signal import savgol_filter
def keypoints_from_signal(x, fs, win=101, poly=3, prom=0.05):
    # smooth
    if win >= len(x): win = len(x)-1 if (len(x)-1)%2==1 else len(x)-2
    if win < 5: win = 5
    xs = savgol_filter(x, win, poly)
    # rising zero crossings
    zc = np.where((xs[:-1] < 0) & (xs[1:] >= 0))[0] + 1
    pk, _ = find_peaks(xs, prominence=prom)
    tr, _ = find_peaks(-xs, prominence=prom)
    idx = np.unique(np.concatenate([zc, pk, tr]))
    return np.sort(idx), xs

# extract ground-based keypoints (we send values at these times)
kp_idx_g = []
for ch in range(6):
    idxg, _ = keypoints_from_signal(gx[ch], fs)
    kp_idx_g.append(idxg)

# counts printed
counts = [len(kp_idx_g[ch]) for ch in range(6)]
print("Keypoint counts per channel (ground selection):")
for ch in range(6):
    print(f"{names[ch]:10s}: {counts[ch]} pts ( {counts[ch]/T:.1f} pts/s )")

# ------- detect anomaly using keypoints (value difference threshold based on baseline MAD) -------
flag_kp = [None]*6
for ch in range(6):
    idx = kp_idx_g[ch]
    v_g = gx[ch, idx]; v_a = air2[ch, idx]
    diff = v_a - v_g
    base_mask = idx < int(5*fs)
    if base_mask.sum()>0:
        med = np.median(diff[base_mask]); mad = np.median(np.abs(diff[base_mask]-med))+1e-12
    else:
        med = np.median(diff); mad = np.median(np.abs(diff-med))+1e-12
    thr = med + 6.0*mad
    f = (np.abs(diff) > thr).astype(int)
    flag_kp[ch] = {'idx': idx, 'diff': diff, 'flag': f, 'thr': thr}

# collect flagged events (payload entries)
events = []
for ch in range(6):
    idxs = flag_kp[ch]['idx'][flag_kp[ch]['flag']==1]
    diffs = flag_kp[ch]['diff'][flag_kp[ch]['flag']==1]
    for i, kpos in enumerate(idxs):
        events.append({'ch': ch, 'kpos': int(kpos), 'amp_d': float(diffs[i])})
print("\nTotal flagged keypoint events (to pack) =", len(events))

# ------- PACK into 32-bit words (bit-field) -------
# Layout (bits 31..0):
# bits31-29: ch (3 bits)
# bits28-13: cycle_idx (16 bits)  <-- we'll use keypoint index as cycle identifier (fits 16 bits)
# bits12-7: amp_q (6 bits, signed two's complement)
# bits6-1: off_q (6 bits, signed)  (we don't send off separately here; we just use amp_d as example)
# bit0: flag (1 bit, set=1)
#
# Because we only have amp_d per kp (difference), we put amp_d in the amp_q field and set off_q=0.
# We quantize amp_d to 6-bit signed (-32..31) using dynamic scale.

# compute scale for amp quantization using max observed absolute amp_d across events
if len(events)==0:
    print("No events to pack.")
    packed = np.array([], dtype=np.uint32)
else:
    max_abs = max(abs(e['amp_d']) for e in events)
    # choose scale so largest maps to +-31; avoid divide by zero
    if max_abs <= 0:
        scale_amp = 1.0
    else:
        scale_amp = 31.0 / max_abs
    # pack
    packed = np.zeros(len(events), dtype=np.uint32)
    t0 = time.perf_counter()
    for i,e in enumerate(events):
        ch = e['ch'] & 0x7
        idx = e['kpos'] & 0xFFFF
        # quantize amp
        q = int(np.round(e['amp_d'] * scale_amp))
        q = max(-32, min(31, q))  # 6-bit signed
        if q < 0:
            q_u = (q + 64) & 0x3F
        else:
            q_u = q & 0x3F
        off_u = 0
        flag = 1
        word = (ch << 29) | (idx << 13) | (q_u << 7) | (off_u << 1) | (flag & 1)
        packed[i] = np.uint32(word)
    pack_time = time.perf_counter() - t0
    packed_bytes = packed.nbytes

    print(f"\nPacked {len(packed)} events into {packed_bytes} bytes (32-bit words). pack_time = {pack_time*1000:.2f} ms")
    print(f"Quantization: scale_amp = {scale_amp:.3f} (amp_d * scale -> q in -32..31)")

# ------- RAW send bytes for comparison -------
# Define raw send as sending the full anomaly windows' raw samples (float32)
# Find combined anomaly window across all channels to be conservative OR per-channel windows
raw_bytes = 0
raw_windows = []
for ch,s0,L,typ,val in anoms:
    si = int(s0*fs); ei = min(N,int((s0+L)*fs))
    raw_windows.append((ch,si,ei))
    raw_bytes += (ei - si) * 4  # 4 bytes per float sample (single channel)
# optionally multiply by number of channels if sending all channels' windows
# For fairness compute bytes to send raw samples for each affected channel:
raw_bytes_total = raw_bytes  # for all channels windows sum of bytes (per-channel samples)
print(f"Raw bytes (sending raw samples for anomaly windows across channels): {raw_bytes_total} bytes")

# ------- measure time to "send" (simulate packing/serializing) -------
# measure time to create raw bytearray for all anomaly window samples (float32)
t0 = time.perf_counter()
# gather raw samples into a float32 array
raw_list = []
for ch,si,ei in raw_windows:
    raw_list.append(air2[ch, si:ei].astype(np.float32))
if len(raw_list)>0:
    raw_concat = np.concatenate(raw_list)
    raw_bytes_measured = raw_concat.nbytes
else:
    raw_bytes_measured = 0
raw_pack_time = time.perf_counter() - t0
print(f"raw_pack_time = {raw_pack_time*1000:.2f} ms, raw_bytes_measured = {raw_bytes_measured} bytes")

# ------- unpack packed words and reconstruct quantized amp_d values, then apply to reconstruct signal (toy) -------
# For simplicity, we will simulate reconstruction by replacing the keypoint value with quantized value then cubic interp
# Build a mapping from kp index to quantized value per channel
if len(events)>0:
    unpacked = []
    qvals = {}  # map (ch,kpos) -> quantized_amp_value (float)
    for w in packed:
        ch = int((w >> 29) & 0x7)
        idx = int((w >> 13) & 0xFFFF)
        q_u = int((w >> 7) & 0x3F)
        # convert back to signed
        if q_u & 0x20:
            q_signed = q_u - 64
        else:
            q_signed = q_u
        amp_q = q_signed / scale_amp
        qvals[(ch, idx)] = amp_q
    # create quantized-keypoint signal: for each channel, build kp values where flagged replaced with quantized
    recon_q = np.zeros_like(air2)
    mse_q = np.zeros(6)
    for ch in range(6):
        idx = kp_idx_g[ch]
        if len(idx) == 0:
            recon_q[ch] = np.zeros(N)
            mse_q[ch] = np.nan
            continue
        vals = air2[ch, idx].copy()
        # replace flagged kp vals with quantized if present
        for kpos_i, kpos in enumerate(idx):
            key = (ch, int(kpos))
            if key in qvals:
                vals[kpos_i] = qvals[key]
        # interp to full
        if len(idx) < 4:
            recon_q[ch] = np.interp(np.arange(N), np.arange(N), air2[ch])
        else:
            f_interp = interp1d(idx, vals, kind='cubic', fill_value='extrapolate')
            recon_q[ch] = f_interp(np.arange(N))
        mse_q[ch] = np.mean((recon_q[ch] - air2[ch])**2)
else:
    recon_q = np.zeros_like(air2); mse_q = np.zeros(6) + np.nan

# ------- compute recon from full-precision keypoints (baseline) -------
recon_kp = np.zeros_like(air2); mse_kp = np.zeros(6)
for ch in range(6):
    idx = kp_idx_g[ch]
    if len(idx) < 4:
        recon_kp[ch] = np.interp(np.arange(N), np.arange(N), air2[ch])
    else:
        f_interp = interp1d(idx, air2[ch, idx], kind='cubic', fill_value='extrapolate')
        recon_kp[ch] = f_interp(np.arange(N))
    mse_kp[ch] = np.mean((recon_kp[ch] - air2[ch])**2)

# ------- report and plots -------
print("\nMSEs (recon from kp full precision) per channel:")
for ch in range(6):
    print(f"{names[ch]:10s}: mse_kp = {mse_kp[ch]:.6e}, mse_q = {mse_q[ch]:.6e}, kp_count = {len(kp_idx_g[ch])}")

print(f"\nTotal events packed = {len(events)}, packed_bytes = {packed_bytes if len(events)>0 else 0}, raw_bytes = {raw_bytes_total}")
print(f"pack_time = {pack_time*1000:.2f} ms, raw_pack_time = {raw_pack_time*1000:.2f} ms")

# Plot a few useful figures:
plt.figure(figsize=(12,10))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t, gx[ch], 'k', lw=0.7, label='ground')
    ax.plot(t, air2[ch], 'b', lw=0.7, label='air(anom)')
    ax.plot(t, recon_kp[ch], 'orange', lw=1.0, label='recon_kp')
    ax.plot(t, recon_q[ch], 'g', lw=1.0, label='recon_quant' if ch==0 else "")
    idx = kp_idx_g[ch]
    ax.scatter(idx/fs, air2[ch, idx], c='r', s=18)
    # highlight flagged idx
    fk = flag_kp[ch]['flag'].astype(bool)
    if fk.sum()>0:
        ax.scatter((idx[fk])/fs, air2[ch, idx[fk]], facecolors='none', edgecolors='m', s=80, linewidths=1.1)
    ax.set_xlim(0, T); ax.set_ylabel(names[ch])
    if ch==0: ax.legend(loc='upper right')
plt.suptitle('Ground (black), Air (blue), Recon from KP (orange), Recon from quantized KP (green); red=kp; magenta circled=flagged')
plt.xlabel('time (s)'); plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()
  
