# Script: two parts in one - (1) 12-keypoint compression pipeline for one signal
# (2) bit-depth slicing analysis for one signal
# Simple variable names, clear prints, plots.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import time

np.random.seed(1)

# ---------------------- Part 1: 12-keypoint pipeline ----------------------
fs = 100                # samples/sec
T = 10                  # seconds window (shorter for clarity)
N = int(fs * T)
t = np.arange(N) / fs

# simple sine signal
f = 1.0                 # Hz
a = 1.0
sig_ground = a * np.sin(2 * np.pi * f * t)

# add small measurement noise to both (air slightly noisier)
sig_air = sig_ground + 0.01 * np.random.randn(N)
sig_ground = sig_ground + 0.005 * np.random.randn(N)  # ground small noise

# initial diff should be ~0
diff_init = sig_air - sig_ground

# inject anomaly: step offset between 4.0s and 4.3s
s0 = 4.0
dur = 0.3
si = int(s0 * fs); ei = int((s0 + dur) * fs)
sig_air_anom = sig_air.copy()
sig_air_anom[si:ei] += 1.5  # offset jump

diff_anom = sig_air_anom - sig_ground

# Plot 1: initial (first 2s) to show flat diff
plt.figure(figsize=(10,4))
plt.plot(t[:200], sig_air[:200], label='air (no anomaly)')
plt.plot(t[:200], sig_ground[:200], label='ground')
plt.title('Initial: air vs ground (first 2s) - should be nearly identical')
plt.xlabel('time (s)'); plt.legend(); plt.grid(True)
plt.show()

plt.figure(figsize=(10,3))
plt.plot(t[:200], diff_init[:200])
plt.title('Initial residual (air - ground) first 2s (near zero)')
plt.xlabel('time (s)'); plt.grid(True); plt.show()

# Plot 2: full signal around anomaly, show square pulse region and residual
plt.figure(figsize=(10,4))
plt.plot(t, sig_air_anom, label='air (with anomaly)')
plt.plot(t, sig_ground, label='ground')
plt.title('Air vs Ground (full signal) with anomaly')
plt.xlabel('time (s)'); plt.legend(); plt.grid(True)
plt.show()

plt.figure(figsize=(10,3))
plt.plot(t, diff_anom, color='red', label='residual (air - ground)')
# overlay ideal square pulse for anomaly region
pulse = np.zeros(N); pulse[si:ei] = 1.0
plt.fill_between(t, 0, pulse * np.max(diff_anom) * 0.9, color='orange', alpha=0.3, label='ideal anomaly window')
plt.title('Residual (air - ground) with anomaly region highlighted')
plt.xlabel('time (s)'); plt.legend(); plt.grid(True); plt.show()

# Extract keypoints (ground-based): smoothed zero-crossings (rising) + peaks + troughs
def keypoints(x, fs, win=101, prom=0.03):
    # simple smoothing
    if win >= len(x):
        win = len(x)-1 if (len(x)-1)%2==1 else len(x)-2
    win = max(5, win)
    xs = savgol_filter(x, win, polyorder=3)
    zc = np.where((xs[:-1] < 0) & (xs[1:] >= 0))[0] + 1
    pk, _ = find_peaks(xs, prominence=prom)
    tr, _ = find_peaks(-xs, prominence=prom)
    idx = np.unique(np.concatenate([zc, pk, tr]))
    return np.sort(idx), xs

idx_g, xs_g = keypoints(sig_ground, fs, win=201, prom=0.02)
# enforce exactly 12 keypoints by uniform thinning if needed
kp_target = 12
if len(idx_g) >= kp_target:
    step = max(1, len(idx_g) // kp_target)
    idx_sel = idx_g[::step][:kp_target]
else:
    # if fewer than target, keep all
    idx_sel = idx_g.copy()

# sample values at selected keypoints (these are what we would send)
vals_send = sig_air_anom[idx_sel]   # sending air values at ground keypoint times
times_send = idx_sel / fs

# detection using only keypoints: compare air vs ground at idx_sel
diff_kp = sig_air_anom[idx_sel] - sig_ground[idx_sel]
# baseline stats from early region (before 1s)
base_mask = idx_sel < int(1.0 * fs)
if base_mask.sum() > 0:
    med = np.median(diff_kp[base_mask]); mad = np.median(np.abs(diff_kp[base_mask] - med)) + 1e-12
else:
    med = np.median(diff_kp); mad = np.median(np.abs(diff_kp-med)) + 1e-12
thr = med + 6 * mad
flag_kp = (np.abs(diff_kp) > thr).astype(int)

# expand flagged keypoints to small windows for visualization
flag_window = np.zeros(N)
halfw = int(0.5 * fs * (1.0 / f))  # half period approx
for i, k in enumerate(idx_sel):
    if flag_kp[i]:
        st = max(0, k-halfw); en = min(N, k+halfw)
        flag_window[st:en+1] = 1

# recon from selected points by cubic interp
if len(idx_sel) >= 4:
    f_interp = interp1d(idx_sel, vals_send, kind='cubic', fill_value='extrapolate')
    recon = f_interp(np.arange(N))
else:
    recon = np.interp(np.arange(N), idx_sel, vals_send)

# Compute bytes: raw vs compressed
# raw: number of samples in anomaly window * 4 bytes (float32)
raw_bytes = (ei - si) * 4
# compressed: number of keypoints sent * 4 bytes (32-bit per point, amplitude only, per choice A)
comp_bytes = len(idx_sel) * 4
# Also compute pack times (simulate by simple operations)
t0 = time.perf_counter()
# simulate raw pack (concatenate samples)
raw_pack = np.concatenate([sig_air_anom[si:ei].astype(np.float32)])
raw_pack_time = (time.perf_counter() - t0)

t0 = time.perf_counter()
# simulate comp pack (pack values_send into bytes)
comp_pack = np.array(vals_send, dtype=np.float32).tobytes()
comp_pack_time = (time.perf_counter() - t0)

# Print summary
print("----- Part1: 12-keypoint pipeline summary -----")
print(f"Total samples N = {N}, fs = {fs} Hz, duration = {T}s")
print(f"Anomaly window samples: start={si}, end={ei}, count={ei-si} samples")
print(f"Raw bytes to send anomaly window = {raw_bytes} bytes")
print(f"Keypoints selected = {len(idx_sel)} (target 12) -> comp bytes = {comp_bytes} bytes")
print(f"Raw pack time = {raw_pack_time*1000:.3f} ms, Comp pack time = {comp_pack_time*1000:.3f} ms")
print(f"Keypoint indices (samples): {idx_sel.tolist()}")
print(f"Keypoint times (s): {[round(x,3) for x in times_send.tolist()]}")
print(f"Keypoint diffs at those times: {[round(x,4) for x in diff_kp.tolist()]}")
print(f"Flags at KP (1=anomaly): {flag_kp.tolist()}")

# Plots for Part1: show keypoints, recon, flags
plt.figure(figsize=(10,4))
plt.plot(t, sig_ground, color='k', lw=0.8, label='ground')
plt.plot(t, sig_air_anom, color='b', lw=0.8, label='air (anom)')
plt.scatter(times_send, vals_send, color='r', s=30, label='kp sent')
plt.title('Part1: Ground (black) vs Air (blue) with KP (red dots)')
plt.xlabel('time (s)'); plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,3))
plt.plot(t, sig_air_anom - sig_ground, color='r', label='residual (full)')
plt.fill_between(t, 0, flag_window * np.max(sig_air_anom - sig_ground) * 0.9, color='orange', alpha=0.3, label='detected region (from KP)')
plt.title('Residual and detected region (KP detection)'); plt.xlabel('time (s)'); plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(10,3))
plt.plot(t, sig_air_anom, label='air (anom)')
plt.plot(t, recon, '--', label='recon from KP')
plt.title('Reconstruction from KP (cubic interp if enough KP)'); plt.xlabel('time (s)'); plt.legend(); plt.grid(True); plt.show()

# ASCII bar representation per point out of 32 bits (4 bytes)
print("\nPer-point 32-bit visualization (each '|' ~ 4 bytes):")
for i, v in enumerate(vals_send):
    bars = '|' * 1  # each point occupies one 32-bit slot (4 bytes)
    print(f"Point {i+1:02d}: {bars}  (4 bytes) value={v:.4f) if False else ''}")

# ---------------------- Part 2: Bit-depth slicing for one signal ----------------------
# Single-signal analysis on same sig_air_anom and sig_ground
bits_list = [32, 16, 8, 6, 4]   # bits to test (32 we'll treat as float32)
results = []

# anomaly detection window threshold baseline (use pre-anomaly 0..3.5s)
base_end = int(3.5 * fs)
for b in bits_list:
    if b == 32:
        # no quantization, use float32 as baseline
        q_air = sig_air_anom.astype(np.float32)
        q_gx = sig_ground.astype(np.float32)
    else:
        levels = 2 ** b
        # map signal to unsigned in range [0, levels-1]
        mn = min(sig_ground.min(), sig_air_anom.min())
        mx = max(sig_ground.max(), sig_air_anom.max())
        # avoid zero range
        if mx - mn < 1e-6:
            q_air = np.zeros_like(sig_air_anom)
            q_gx = np.zeros_like(sig_ground)
        else:
            qa = np.round((sig_air_anom - mn) / (mx - mn) * (levels - 1)).astype(int)
            qg = np.round((sig_ground - mn) / (mx - mn) * (levels - 1)).astype(int)
            # reconstruct back to amplitude domain for residual detection
            q_air = (qa.astype(float) / (levels - 1)) * (mx - mn) + mn
            q_gx  = (qg.astype(float) / (levels - 1)) * (mx - mn) + mn
    # compute residual and detection: threshold med+6*MAD on baseline
    res = q_air - q_gx
    med = np.median(res[:base_end]); mad = np.median(np.abs(res[:base_end] - med)) + 1e-12
    thr = med + 6 * mad
    # detection if any residual during anomaly window exceeds thr in abs value
    detected = np.any(np.abs(res[si:ei]) > thr)
    results.append({'bits': b, 'detected': bool(detected), 'thr': thr})

# Print bit-depth results
print("\n--- Bit-depth detection results (unsigned quantization mapping) ---")
for r in results:
    b = r['bits']
    det = r['detected']
    print(f"{b:2d} bits: detected anomaly? -> {det} ; bytes/point = {b//8}")

# Plot quantized signals for visual inspection (show a zoom around anomaly)
zoom_s = int((s0 - 1.0) * fs); zoom_e = int((s0 + dur + 1.0) * fs)
plt.figure(figsize=(10,8))
for i, b in enumerate(bits_list):
    plt.subplot(len(bits_list), 1, i+1)
    if b == 32:
        q = sig_air_anom
        gxq = sig_ground
    else:
        levels = 2 ** b
        mn = min(sig_ground.min(), sig_air_anom.min())
        mx = max(sig_ground.max(), sig_air_anom.max())
        qa = np.round((sig_air_anom - mn) / (mx - mn) * (levels - 1)).astype(int)
        q = (qa.astype(float) / (levels - 1)) * (mx - mn) + mn
        qg = np.round((sig_ground - mn) / (mx - mn) * (levels - 1)).astype(int)
        gxq = (qg.astype(float) / (levels - 1)) * (mx - mn) + mn
    plt.plot(t[zoom_s:zoom_e], q[zoom_s:zoom_e], label=f'air ({b} bits)')
    plt.plot(t[zoom_s:zoom_e], gxq[zoom_s:zoom_e], '--', label=f'ground ({b} bits)')
    plt.title(f'{b} bits - zoom around anomaly')
    if i==0: plt.legend()
plt.xlabel('time (s)'); plt.tight_layout(); plt.show()

# Print final recommendation:
min_bits = min([r['bits'] for r in results if r['detected']])
print(f"\nMinimum bits that still detected anomaly in this test: {min_bits} bits per point")

