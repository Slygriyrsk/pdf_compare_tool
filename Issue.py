"""
keypoint_anom.py
- 6 sensors sine signals
- inject step anomalies (different start times)
- add noise
- extract keypoints: zero-crossings, local peaks, local troughs
- compute residual at keypoints -> binary flags (1 during anomaly)
- reconstruct simple waveform from keypoints (interp) for visualization
- show FFT bins (diagnostic)
- payload: for each flagged region send {ch, start_idx, dur, off_est}
Short var names for easy debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft

np.random.seed(0)

# ----setup----
fs = 10          # samples per sec -> 0.1s interval
N = 30           # number of samples
t = np.arange(N) / fs

# 6 channels: frequencies, amplitudes
f = np.array([0.5, 1.0, 1.5, 0.7, 1.2, 0.9])   # Hz
A = np.array([1.0, 0.8, 0.6, 1.1, 0.9, 0.7])

# clean traces (air and ground copy identical initially)
air_clean = np.zeros((6, N))
for i in range(6):
    air_clean[i] = A[i] * np.sin(2*np.pi*f[i]*t)

gx_clean = air_clean.copy()   # ground copy (ideal baseline)

# ----noise----
sigma_air = 0.05   # measurement noise in-air
sigma_gx  = 0.02   # measurement noise ground baseline
air = air_clean + sigma_air * np.random.randn(6, N)
gx  = gx_clean  + sigma_gx  * np.random.randn(6, N)

# ----inject STEP anomalies----
# start indices (0-based) and offsets for each channel
sidx = [6, 9, 12, 15, 18, 21]   # start sample index for each channel
dur = 6                         # anomaly duration in samples
offs = [0.8, -1.0, 0.5, -0.6, 1.2, -0.9]

for ch in range(6):
    si = sidx[ch]
    ei = min(N, si + dur)
    air[ch, si:ei] += offs[ch]

# ----keypoint extraction function----
def get_keypoints(x):
    """
    Given a 1D array x of length N:
    - find zero-crossing indices (first index where sign changes)
    - find local maxima (peaks)
    - find local minima (troughs) by inverted peaks
    Return sorted unique indices.
    """
    # zero-crossings: index i where x[i]*x[i+1] < 0 -> consider i or i+1
    zc = np.where(np.diff(np.sign(x)) != 0)[0]
    # use i+1 as the crossing point (more stable)
    zc = zc + 1
    # peaks
    pk_idx, _ = find_peaks(x)
    # troughs (peaks of -x)
    tr_idx, _ = find_peaks(-x)
    idx = np.unique(np.concatenate((zc, pk_idx, tr_idx)))
    # clamp to valid indices
    idx = idx[(idx >= 0) & (idx < len(x))]
    return np.sort(idx)

# ----collect keypoints for all channels----
kp_idx = [None]*6
for ch in range(6):
    kp_idx[ch] = get_keypoints(air[ch])

# Ensure minimal coverage: if kp list is empty (rare), sample 0 and N-1
for ch in range(6):
    if kp_idx[ch].size == 0:
        kp_idx[ch] = np.array([0, N-1])

# ----compute residual only at keypoints and detect anomaly----
thr = 0.1   # threshold for flagging residual at keypoint (tuneable)
flag_kp = [None]*6           # boolean array per channel for each kp
flag_full = np.zeros((6, N), dtype=int)  # final binary alarm per sample (square wave)

payload = []  # minimal payload entries

for ch in range(6):
    idx = kp_idx[ch]
    # values at keypoints
    v_air = air[ch, idx]
    v_gx  = gx[ch, idx]
    v_r   = v_air - v_gx
    # flag kp if abs residual > thr
    f_k = np.abs(v_r) > thr
    flag_kp[ch] = f_k.astype(int)

    # Convert kp flags to full-sample square wave: find contiguous region around flagged kp
    # Heuristic: if any kp within [si, ei) flagged, mark whole [si, ei) as anomalous where si and ei from sidx/dur
    # But we don't assume knowledge of sidx in general. Instead we:
    # - map each flagged kp index to nearest sample index (since kp are indices already)
    # - expand a small window around that sample (half window size = floor(dur/2))
    halfw = max(1, dur//3)
    for kpos, fk in zip(idx, f_k):
        if fk:
            st = max(0, kpos - halfw)
            en = min(N-1, kpos + halfw)
            flag_full[ch, st:en+1] = 1

    # Combine contiguous flagged regions to form payload entries
    vec = flag_full[ch]
    in_r = False
    for s in range(N):
        if (not in_r) and (vec[s] == 1):
            in_r = True
            st = s
        if in_r and ((s == N-1) or (vec[s] == 1 and vec[min(s+1, N-1)] == 0)):
            en = s
            in_r = False
            dur_samps = en - st + 1
            off_est = float(np.mean(air[ch, st:en+1] - gx[ch, st:en+1]))
            payload.append({'ch': ch, 'si': st, 'dur': dur_samps, 'off': off_est})

# ----reconstruct from keypoints for visualization----
# Simple linear interpolation using keypoints (air-side points). Then ground adds reconstructed delta.
recon = gx.copy()   # baseline
for ch in range(6):
    idx = kp_idx[ch]
    xs = idx
    ys = air[ch, xs]
    # If only one point, fill constant; else linear interp
    if xs.size == 1:
        est = np.ones(N) * ys[0]
    else:
        est = np.interp(np.arange(N), xs, ys)
    # reconstruct delta as est - gx_est (approx); but we want baseline + delta --> show approx air
    recon[ch] = est  # approximate full waveform reconstructed from keypoints

# ----FFT diagnostics (full-signal FFT)----
X_air = np.abs(fft(air, axis=1))[:, :N//2]
X_gx  = np.abs(fft(gx, axis=1))[:, :N//2]
# find dominant bin change in anomaly region: (for display only)
dom_bin = np.argmax(X_air, axis=1)

# ----plots: show signals, keypoints, residual kp flags, square flags, FFT bins----
plt.rcParams.update({'figure.max_open_warning': 0})

# plot 6 channels: time-domain with keypoints
plt.figure(figsize=(10,12))
for ch in range(6):
    ax = plt.subplot(6,1,ch+1)
    ax.plot(t, air[ch], label='air', linewidth=1)
    ax.plot(t, gx[ch], label='gx', linewidth=0.8)
    kx = kp_idx[ch]
    ax.scatter(kx / fs, air[ch, kx], color='r', s=30, zorder=5, label='kp' if ch==0 else "")
    ax.set_xlim(0, (N-1)/fs)
    ax.set_ylabel(f's{ch}')
    if ch == 0:
        ax.legend(loc='upper right')
plt.xlabel('time (s)')
plt.suptitle('Air vs Ground with keypoints (red dots)')
plt.tight_layout(rect=[0,0.03,1,0.95])

# plot residual at keypoints binary map
plt.figure(figsize=(10,6))
for ch in range(6):
    idx = kp_idx[ch]
    flags = flag_kp[ch]
    # map kp positions to time and plot stem of flags
    plt.subplot(6,1,ch+1)
    plt.stem(idx / fs, flags, use_line_collection=True)
    plt.ylim(-0.1, 1.1)
    plt.ylabel(f's{ch}')
    if ch==0:
        plt.title('Binary flag at keypoints (1 = anomaly at that kp)')
plt.xlabel('time (s)')
plt.tight_layout(rect=[0,0.03,1,0.95])

# square-wave (full) flags per channel (visual)
plt.figure(figsize=(10,6))
for ch in range(6):
    plt.subplot(6,1,ch+1)
    plt.step(t, flag_full[ch], where='post')
    plt.ylim(-0.1, 1.1)
    plt.ylabel(f's{ch}')
    if ch==0:
        plt.title('Square-wave flags (expanded from flagged keypoints)')
plt.xlabel('time (s)')
plt.tight_layout(rect=[0,0.03,1,0.95])

# FFT diagnostic heatmap (dominant bins)
plt.figure(figsize=(8,4))
plt.imshow(X_air, aspect='auto', origin='lower', extent=[0, fs/2, 0, 6])
plt.colorbar(label='mag')
plt.yticks(np.arange(6)+0.5, [f's{i}' for i in range(6)])
plt.xlabel('freq (Hz)')
plt.title('FFT mag (air) per channel (first N/2 bins)')
plt.tight_layout()

# print payloads and keypoint counts
print("Keypoints per channel (indices):")
for ch in range(6):
    print(f"ch{ch}: {kp_idx[ch].tolist()} (count={len(kp_idx[ch])})")

print("\nPayload entries (ch, start_idx, duration_samps, offset_est):")
for p in payload:
    print(p)

plt.show()
  
