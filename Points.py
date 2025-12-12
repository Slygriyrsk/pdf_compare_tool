# cycle_feature_demo.py
# Usage: python cycle_feature_demo.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
np.random.seed(0)

# ---------- params ----------
fs = 50              # samples per sec (0.02s)
T = 30               # seconds (your requested period)
N = int(fs * T)
t = np.arange(N)/fs

names = ['roll','pitch','yaw','roll_rate','pitch_rate','yaw_rate']
freqs = np.array([0.5, 0.7, 1.1, 0.6, 0.9, 1.3])   # Hz
amps  = np.array([1.0, 1.2, 0.9, 1.1, 0.95, 0.8])

# ---------- generate signals ----------
air_clean = np.zeros((6,N))
for i in range(6):
    air_clean[i] = amps[i] * np.sin(2*np.pi*freqs[i]*t)

gx_clean = air_clean.copy()

# add measurement noise (air > ground)
sig_air = 0.02
sig_gx  = 0.008
air = air_clean + sig_air * np.random.randn(6,N)
gx  = gx_clean + sig_gx  * np.random.randn(6,N)

# ---------- initial plots: first 5s (air vs ground + residual) ----------
zoom0 = 0; zoom1 = 5.0
zi = int(zoom0*fs); ze = int(zoom1*fs)

plt.figure(figsize=(12,9))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t[zi:ze], air[ch,zi:ze], label='air', lw=0.9)
    ax.plot(t[zi:ze], gx[ch,zi:ze], label='ground', lw=0.8)
    ax.set_ylabel(names[ch])
    if ch==0: ax.legend()
plt.suptitle('Initial: air vs ground (first 5 s) - near identical (noise only)')
plt.xlabel('time (s)')
plt.tight_layout(rect=[0,0.03,1,0.95])

plt.figure(figsize=(10,5))
for ch in range(6):
    plt.plot(t[zi:ze], air[ch,zi:ze] - gx[ch,zi:ze], label=names[ch])
plt.title('Initial residuals (air - ground), first 5 s (should be ~0)')
plt.xlabel('time (s)'); plt.legend(); plt.tight_layout()

# ---------- inject anomalies (step offsets) ----------
anoms = [
    (0, 10.0, 0.6, 'offset', 1.5),
    (1, 15.0, 0.8, 'offset', -1.2),
    (2, 20.0, 1.0, 'offset', 2.0),
    (3, 8.5, 0.4, 'offset', 0.9),
    (4, 12.0, 0.7, 'offset', -1.6),
    (5, 22.0, 1.2, 'offset', 1.2),
]
air2 = air.copy()
for ch,s0,L,typ,val in anoms:
    si = int(s0*fs); ei = min(N, int((s0+L)*fs))
    if typ == 'offset':
        air2[ch, si:ei] += val

res_full = air2 - gx

# ---------- show anomaly windows and residuals (with ideal square highlight) ----------
plt.figure(figsize=(12,14))
for ch in range(6):
    s0 = anoms[ch][1]; L = anoms[ch][2]
    zi = int(max(0,(s0-2.0)*fs)); ze = int(min(N,(s0+L+2.0)*fs))
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t[zi:ze], air2[ch,zi:ze], label='air', lw=0.9)
    ax.plot(t[zi:ze], gx[ch,zi:ze], label='ground', lw=0.8)
    ax.set_ylabel(names[ch])
    ax2 = ax.twinx()
    res_seg = res_full[ch, zi:ze]
    ax2.plot(t[zi:ze], res_seg, color='red', alpha=0.6, lw=0.9, label='residual')
    si = int(anoms[ch][1]*fs); ei = min(N, int((anoms[ch][1]+anoms[ch][2])*fs))
    pulse = np.zeros(ze-zi)
    if ei>zi and si<ze:
        pst = max(0, si-zi); pen = min(ze-zi, ei-zi)
        pulse[pst:pen] = 1.0
        ax2.fill_between(t[zi:ze], 0, pulse*np.max(res_seg)*1.1, color='orange', alpha=0.25)
    if ch==0: ax.legend(loc='upper left'); ax2.legend(loc='upper right')
plt.suptitle('After anomaly injection: air vs ground (residual red). Orange = ideal anomaly region')
plt.xlabel('time (s)'); plt.tight_layout(rect=[0,0.03,1,0.95])

# ---------- keypoint extraction (ground-based points to send) ----------
from scipy.signal import savgol_filter
def keypoints_from_signal(x, fs, win=101, poly=3):
    # smooth (Savitzky-Golay) then find rising zero crossings, peaks and troughs (prominence threshold)
    if win >= len(x): 
        win = len(x)-1 if (len(x)-1)%2==1 else len(x)-2
    if win < 5: win = 5
    xs = savgol_filter(x, win, poly)
    zc = np.where((xs[:-1] < 0) & (xs[1:] >= 0))[0] + 1
    pk, _ = find_peaks(xs, prominence=0.05)
    tr, _ = find_peaks(-xs, prominence=0.05)
    idx = np.unique(np.concatenate([zc, pk, tr]))
    return np.sort(idx), xs

kp_idx_ground = []
kp_idx_air = []
for ch in range(6):
    idx_g, xs_g = keypoints_from_signal(gx[ch], fs)
    idx_a, xs_a = keypoints_from_signal(air2[ch], fs)
    kp_idx_ground.append(idx_g)
    kp_idx_air.append(idx_a)

counts = [len(kp_idx_ground[ch]) for ch in range(6)]
print("Keypoint counts per channel (ground-based selection):")
for ch in range(6):
    print(f"{names[ch]:10s}: {counts[ch]} points over {T}s -> {counts[ch]/T:.1f} pts/s")

# ---------- detect anomaly using only keypoints ----------
flag_kp = [None]*6
for ch in range(6):
    idx = kp_idx_ground[ch]
    v_g = gx[ch, idx]; v_a = air2[ch, idx]
    diff = v_a - v_g
    base_mask = idx < int(5*fs)
    if base_mask.sum()>0:
        med = np.median(diff[base_mask]); mad = np.median(np.abs(diff[base_mask]-med)) + 1e-12
    else:
        med = np.median(diff); mad = np.median(np.abs(diff-med))+1e-12
    thr = med + 6*mad
    f = (np.abs(diff) > thr).astype(int)
    flag_kp[ch] = {'idx': idx, 'diff': diff, 'flag': f, 'thr': thr}

print("\nDetection summary (first flagged time and count per channel):")
for ch in range(6):
    idxs = flag_kp[ch]['idx'][flag_kp[ch]['flag']==1]
    if idxs.size>0:
        print(names[ch], "first flag at", idxs[0]/fs, "s ; count =", idxs.size)
    else:
        print(names[ch], "no flag (on KP)")

# ---------- zero-crossing-only extreme test ----------
zc_counts = []
zc_map = []
for ch in range(6):
    zc = np.where((gx[ch][:-1] < 0) & (gx[ch][1:] >= 0))[0] + 1
    zc_counts.append(len(zc)); zc_map.append(zc)
print("\nZero-crossing counts (rising) per channel:", zc_counts)

# ---------- reconstruct from keypoints (cubic interp) and compute MSE ----------
from scipy.interpolate import interp1d
recon_from_kp = np.zeros_like(air2)
mse_kp = np.zeros(6)
for ch in range(6):
    idx = kp_idx_ground[ch]
    if len(idx) < 4:
        recon_from_kp[ch] = np.interp(np.arange(N), np.arange(N), air2[ch])
    else:
        f_interp = interp1d(idx, air2[ch, idx], kind='cubic', fill_value='extrapolate')
        recon_from_kp[ch] = f_interp(np.arange(N))
    mse_kp[ch] = np.mean((recon_from_kp[ch] - air2[ch])**2)

# ---------- plotting reconstruction, kp markers, flagged kp ----------
plt.figure(figsize=(14,18))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t, gx[ch], color='k', lw=0.7, label='ground')
    ax.plot(t, air2[ch], color='b', lw=0.7, label='air (anom)')
    ax.plot(t, recon_from_kp[ch], color='orange', lw=1.0, label='recon from kp', alpha=0.9)
    idx = kp_idx_ground[ch]
    ax.scatter(idx/fs, air2[ch, idx], color='r', s=18, label='kp (sent)' if ch==0 else "")
    fk = flag_kp[ch]['flag'].astype(bool)
    if fk.sum()>0:
        ax.scatter((idx[fk])/fs, air2[ch, idx[fk]], facecolors='none', edgecolors='m', s=80, linewidths=1.1, label='kp flagged' if ch==0 else "")
    ax.set_xlim(0, T); ax.set_ylabel(names[ch])
    if ch==0: ax.legend(loc='upper right')
plt.suptitle('Air vs Ground vs Recon from Keypoints (red=keypoints sent; magenta circles=flagged KP)')
plt.xlabel('time (s)'); plt.tight_layout(rect=[0,0.03,1,0.95])

# ---------- residual + detected regions (expanded from flagged kps) ----------
plt.figure(figsize=(12,8))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t, air2[ch]-gx[ch], color='r', label='residual (air-gx)')
    fmask = np.zeros(N)
    idx = flag_kp[ch]['idx'][flag_kp[ch]['flag']==1]
    halfw = int(0.5 * fs / max(0.1, freqs[ch]))
    for kpos in idx:
        st = max(0, kpos-halfw); en = min(N, kpos+halfw)
        fmask[st:en+1] = 1.0
    ax.fill_between(t, 0, fmask*np.max(air2[ch]-gx[ch])*1.05, color='orange', alpha=0.25, label='detected region (kp)' if ch==0 else "")
    ax.set_xlim(0, T); ax.set_ylabel(names[ch])
    if ch==0: ax.legend()
plt.suptitle('Residual and detected regions (from keypoint detection)'); plt.xlabel('time (s)'); plt.tight_layout(rect=[0,0.03,1,0.95])

# ---------- final summaries ----------
print("\nReconstruction MSE from keypoints per channel:")
for ch in range(6):
    print(f"{names[ch]:10s}: mse = {mse_kp[ch]:.6e}, kp_count = {len(kp_idx_ground[ch])}")

total_flagged = sum([int(np.sum(flag_kp[ch]['flag'])) for ch in range(6)])
print("\nTotal flagged keypoint events (payload entries) =", total_flagged)
print("Total keypoints (if sending all ground keypoints) =", sum([len(kp_idx_ground[ch]) for ch in range(6)]))

plt.show()
