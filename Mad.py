# Implementation of Option D: Cycle-Feature Encoding (Amplitude, Phase, Offset)
# - Simulate 6 signals (30s), sample at fs
# - Inject step anomalies for some cycles
# - Extract cycles via zero-crossings, compute per-cycle features:
#     amp = max - min
#     offset = (max + min)/2
#     phase = time between successive zero crossings (period)
# - Compare air vs ground features, compute differences
# - Detect anomalies by thresholding feature diffs (amplitude and offset)
# - Plot: signals, cycle boundaries, per-cycle features, diffs, detection flags
# Short variable names and clear section headers as comments.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft

np.random.seed(0)

# ----params----
fs = 50            # sampling frequency Hz
T = 30             # total duration seconds
N = int(T * fs)
t = np.arange(N) / fs

# base frequencies and amplitudes for 6 sensors (Hz)
f = np.array([0.5, 0.7, 1.1, 0.6, 0.9, 1.3])
A = np.array([1.0, 1.2, 0.9, 1.1, 0.95, 0.8])

# ----generate clean signals----
air_clean = np.zeros((6, N))
for i in range(6):
    air_clean[i] = A[i] * np.sin(2*np.pi*f[i]*t)

gx_clean = air_clean.copy()

# ----add noise (realistic)----
sig_air = 0.03
sig_gx = 0.01
air = air_clean + sig_air * np.random.randn(6, N)
gx  = gx_clean + sig_gx  * np.random.randn(6, N)

# ----inject anomalies (step offsets or amplitude change per cycle)----
# We'll inject anomalies in a few sensors at certain time ranges (in samples)
anom_specs = [
    (0, int(10*fs), int(0.6*fs), 'offset', 1.5),   # ch 0, start 10s, duration 0.6s, offset +1.5
    (1, int(15*fs), int(0.8*fs), 'amp', 1.8),      # ch1 amplitude multiply during anomaly region
    (2, int(20*fs), int(1.0*fs), 'offset', -2.0),
    (3, int(8*fs),  int(0.4*fs), 'offset', 1.0),
    (4, int(12*fs), int(0.7*fs), 'amp', 1.5),
    (5, int(22*fs), int(1.2*fs), 'offset', -1.2),
]
for ch, s0, L, typ, val in anom_specs:
    e = min(N, s0+L)
    if typ == 'offset':
        air[ch, s0:e] += val
    elif typ == 'amp':
        air[ch, s0:e] *= val

# ----cycle segmentation via zero-crossings (rising edges)----
# For each channel, find rising zero-crossings indices -> mark cycle starts
zc_idx = []
for ch in range(6):
    x = air[ch]  # use air for zero-crossings to detect cycles (could use ground but air has anomaly)
    # sign change from negative to positive -> rising zero crossing
    zc = np.where((x[:-1] < 0) & (x[1:] >= 0))[0] + 1
    # ensure first cycle starts at 0 if earlier
    if zc.size == 0 or zc[0] != 0:
        # prepend 0 if not present
        zc = np.insert(zc, 0, 0)
    zc_idx.append(zc)

# Also compute ground zc for alignment (use ground signals)
zc_idx_g = []
for ch in range(6):
    x = gx[ch]
    zc = np.where((x[:-1] < 0) & (x[1:] >= 0))[0] + 1
    if zc.size == 0 or zc[0] != 0:
        zc = np.insert(zc, 0, 0)
    zc_idx_g.append(zc)

# ----function to compute per-cycle features----
def cycle_features(x, zc):
    # inputs: x (1D signal), zc (indices of cycle starts)
    # returns arrays of amp, offset, period (seconds), mid_time (sec)
    amps = []
    offs = []
    periods = []
    midt = []
    for i in range(len(zc)-1):
        s = zc[i]
        e = zc[i+1]
        seg = x[s:e]
        if seg.size == 0:
            amps.append(0.0); offs.append(0.0); periods.append((e-s)/fs); midt.append((s+e)/2/fs); continue
        mx = np.max(seg); mn = np.min(seg)
        amps.append(mx - mn)
        offs.append((mx + mn)/2.0)
        periods.append((e - s) / fs)
        midt.append((s + e) / 2.0)
    # handle last partial cycle from last zc to end
    s = zc[-1]; e = len(x)
    seg = x[s:e]
    if seg.size>0:
        mx = np.max(seg); mn = np.min(seg)
        amps.append(mx - mn)
        offs.append((mx + mn)/2.0)
        periods.append((e - s) / fs)
        midt.append((s + e)/2.0)
    return np.array(amps), np.array(offs), np.array(periods), np.array(midt)

# ----compute features for air and ground----
feat_air = [None]*6
feat_gx = [None]*6
for ch in range(6):
    amp_a, off_a, per_a, mt_a = cycle_features(air[ch], zc_idx[ch])
    amp_g, off_g, per_g, mt_g = cycle_features(gx[ch], zc_idx_g[ch])
    feat_air[ch] = {'amp': amp_a, 'off': off_a, 'per': per_a, 't': mt_a}
    feat_gx[ch]  = {'amp': amp_g, 'off': off_g, 'per': per_g, 't': mt_g}

# ----align cycles between air and ground: use cycle mid-times to map closest cycles----
# produce per-cycle comparisons at air cycle midtimes mapping to nearest ground cycle index
cmp = [None]*6
for ch in range(6):
    mt_a = feat_air[ch]['t']
    mt_g = feat_gx[ch]['t']
    amp_a = feat_air[ch]['amp']; off_a = feat_air[ch]['off']; per_a = feat_air[ch]['per']
    amp_g = feat_gx[ch]['amp']; off_g = feat_gx[ch]['off']; per_g = feat_gx[ch]['per']
    # find nearest ground cycle index for each air midtime
    idx_map = np.array([np.argmin(np.abs(mt_g - ta)) if mt_g.size>0 else -1 for ta in mt_a])
    # compute differences for mapped cycles
    amp_diff = np.zeros_like(amp_a); off_diff = np.zeros_like(off_a); per_diff = np.zeros_like(per_a)
    for i, ig in enumerate(idx_map):
        if ig >= 0 and ig < len(amp_g):
            amp_diff[i] = amp_a[i] - amp_g[ig]
            off_diff[i] = off_a[i] - off_g[ig]
            per_diff[i] = per_a[i] - per_g[ig]
        else:
            amp_diff[i] = amp_a[i]; off_diff[i] = off_a[i]; per_diff[i] = per_a[i]
    cmp[ch] = {'mt': mt_a, 'amp_a': amp_a, 'amp_g': amp_g[idx_map] if len(amp_g)>0 else np.zeros_like(amp_a),
               'amp_diff': amp_diff, 'off_a': off_a, 'off_g': off_g[idx_map] if len(off_g)>0 else np.zeros_like(off_a),
               'off_diff': off_diff, 'per_diff': per_diff}

# ----detection: thresholding on amp_diff and off_diff----
# use robust thresholds: median + k * mad of baseline cycles (before 5s)
k_amp = 6.0; k_off = 6.0
flags = [None]*6
payload = []  # list of minimal payload entries: (ch, cycle_mid_time, amp_diff, off_diff)
for ch in range(6):
    mt = cmp[ch]['mt']
    # baseline cycles indices where mid-time < 5s
    base_idx = np.where(mt < 5.0)[0]
    if base_idx.size == 0:
        med_amp = np.median(cmp[ch]['amp_diff'])
        mad_amp = np.median(np.abs(cmp[ch]['amp_diff'] - med_amp))
        med_off = np.median(cmp[ch]['off_diff'])
        mad_off = np.median(np.abs(cmp[ch]['off_diff'] - med_off))
    else:
        med_amp = np.median(cmp[ch]['amp_diff'][base_idx]); mad_amp = np.median(np.abs(cmp[ch]['amp_diff'][base_idx] - med_amp))
        med_off = np.median(cmp[ch]['off_diff'][base_idx]); mad_off = np.median(np.abs(cmp[ch]['off_diff'][base_idx] - med_off))
    thr_amp = med_amp + k_amp * mad_amp
    thr_off = med_off + k_off * mad_off
    # flag cycles where either amp_diff or off_diff exceed thresholds (also consider negative side by abs)
    f = (np.abs(cmp[ch]['amp_diff']) > thr_amp) | (np.abs(cmp[ch]['off_diff']) > thr_off)
    flags[ch] = f.astype(int)
    # create payload entries for flagged cycles
    for i, fi in enumerate(f):
        if fi:
            payload.append({'ch': ch, 't': mt[i], 'amp_d': float(cmp[ch]['amp_diff'][i]), 'off_d': float(cmp[ch]['off_diff'][i])})

# ----reconstruction from features (approx) for visualization----
# reconstruct a simple sine per cycle using ground cycle period and air amp/offset when flagged else ground amp/offset
recon = np.zeros((6, N))
for ch in range(6):
    mt = cmp[ch]['mt']
    amp_g = feat_gx[ch]['amp']; off_g = feat_gx[ch]['off']; per_g = feat_gx[ch]['per']
    # map ground cycles by midtimes to get period per air cycle
    if len(mt)==0:
        recon[ch] = gx[ch]
        continue
    # For each cycle i, determine start/end indices and synthesize sine with chosen amplitude and offset
    # use zc indices for air to get boundaries
    zc = zc_idx[ch]
    for i in range(len(mt)):
        s = zc[i] if i < len(zc) else 0
        e = zc[i+1] if (i+1) < len(zc) else N
        # choose amp and off: if flagged, use air values; else use ground values mapped
        use_amp = cmp[ch]['amp_a'][i] if flags[ch][i]==1 else (feat_gx[ch]['amp'][min(i, len(feat_gx[ch]['amp'])-1)] if len(feat_gx[ch]['amp'])>0 else cmp[ch]['amp_a'][i])
        use_off = cmp[ch]['off_a'][i] if flags[ch][i]==1 else (feat_gx[ch]['off'][min(i, len(feat_gx[ch]['off'])-1)] if len(feat_gx[ch]['off'])>0 else cmp[ch]['off_a'][i])
        # synthesize simple half-cycle sine scaled to use_amp and centered at use_off
        seg_t = np.arange(s, e) / fs
        # approximate frequency from period if available, else fallback to f[ch]
        per = cmp[ch]['per_diff'][i] + (feat_gx[ch]['per'][min(i, len(feat_gx[ch]['per'])-1)] if len(feat_gx[ch]['per'])>0 else 1.0/f[ch])
        per = max(per, 1.0/(f[ch]*2))  # avoid zero
        freq = 1.0 / per if per>0 else f[ch]
        # create sine with amplitude half (since amp = max-min), use_amp/2 as amplitude
        recon[ch, s:e] = use_off + (use_amp/2.0) * np.sin(2*np.pi*freq*seg_t)
        
# ----plots----
plt.rcParams.update({'figure.max_open_warning': 0})

# 1) plot signals for all 6 channels with cycle midpoints and flagged cycles highlighted
plt.figure(figsize=(12,14))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t, gx[ch], label='gx (ground)', color='k', linewidth=0.7)
    ax.plot(t, air[ch], label='air', color='b', linewidth=0.6)
    # plot cycle midpoints
    mt = cmp[ch]['mt']
    ax.scatter(mt, np.zeros_like(mt), marker='v', color='g', s=20, label='cycle mid' if ch==0 else "")
    # highlight flagged cycles as shaded regions
    for i, mm in enumerate(mt):
        if flags[ch][i]==1:
            # shade +/- half period
            hp = (cmp[ch]['per_diff'][i] + (feat_gx[ch]['per'][min(i,len(feat_gx[ch]['per'])-1)] if len(feat_gx[ch]['per'])>0 else 1.0/f[ch]))/2.0
            ax.axvspan(mm-hp, mm+hp, color='r', alpha=0.2)
    ax.set_ylabel(f's{ch}'); ax.set_xlim(0, T)
    if ch==0: ax.legend(loc='upper right')
plt.xlabel('time (s)'); plt.suptitle('Signals: ground (black) vs air (blue). Red shaded = flagged cycles'); plt.tight_layout(rect=[0,0.03,1,0.95])

# 2) per-cycle features and diffs for a sample channel (show all channels in subplots)
plt.figure(figsize=(12,10))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    mt = cmp[ch]['mt']
    ax.plot(mt, cmp[ch]['amp_a'], 'b.-', label='amp_air')
    ax.plot(mt, cmp[ch]['amp_g'], 'k.--', label='amp_gx')
    ax.plot(mt, cmp[ch]['off_a'], 'c.-', label='off_air')
    ax.plot(mt, cmp[ch]['off_g'], 'm.--', label='off_gx')
    ax.plot(mt, cmp[ch]['amp_diff'], 'r.-', label='amp_diff')
    ax.set_ylabel(f's{ch}'); ax.set_xlim(0, T)
    if ch==0: ax.legend(loc='upper right')
plt.xlabel('time (s)'); plt.suptitle('Per-cycle features: amp, offset and amp_diff (red)'); plt.tight_layout(rect=[0,0.03,1,0.95])

# 3) payload summary and detection flags heatmap
plt.figure(figsize=(10,4))
flag_map = np.zeros((6, max(len(cmp[ch]['mt']) for ch in range(6))))
flag_map[:] = np.nan
for ch in range(6):
    L = len(cmp[ch]['mt'])
    flag_map[ch, :L] = flags[ch]
plt.imshow(flag_map, aspect='auto', origin='lower', cmap='viridis', extent=[0, T, 0, 6])
plt.colorbar(label='flag (1=anomaly)'); plt.yticks(np.arange(6)+0.5, [f's{ch}' for ch in range(6)]); plt.xlabel('time (s)'); plt.title('Flag map (per-cycle)'); plt.tight_layout()

# 4) show reconstructed approx signal from features overlayed on actual air for a channel example (ch 0..5)
plt.figure(figsize=(12,8))
for ch in range(6):
    ax = plt.subplot(3,2,ch+1)
    ax.plot(t, air[ch], label='air', linewidth=0.6)
    ax.plot(t, recon[ch], label='recon', linewidth=0.8)
    ax.set_title(f'ch{ch}'); ax.set_xlim(0, T)
    if ch==0: ax.legend()
plt.suptitle('Air signal (blue) vs recon from features (orange)'); plt.tight_layout(rect=[0,0.03,1,0.95])

# 5) print payload byte-size estimate
# Each payload entry contains: ch (1 byte), t (float 4 bytes), amp_d (float4), off_d (float4) -> 13 bytes approx -> align to 16 bytes
payload_count = len(payload)
bytes_per_entry = 16
total_bytes = payload_count * bytes_per_entry
print(f"Payload entries: {payload_count}, approx bytes (aligned): {total_bytes} bytes")

# list some payload entries
print("\nSample payload entries (first 10):")
for p in payload[:10]:
    print(p)

plt.show()
