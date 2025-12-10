# ----setup---- Code
# Simulate 6 signals, add noise, inject STEP anomalies, then compare full sampling vs sparse sampling
# Show detection, payload sizes, and plots. Variables are short for easy debugging.
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
np.random.seed(42)

# params
fs = 10        # 10 samples/sec -> 0.1s interval
N = 30         # total samples
t = np.arange(N) / fs

# 6 channels s0..s5
f = np.array([0.5, 1.0, 1.5, 0.7, 1.2, 0.9])
A = np.array([1.0, 0.8, 0.6, 1.1, 0.9, 0.7])
ph = 2*np.pi*np.random.rand(6)

# base clean signals
tx_clean = np.zeros((6, N))
for i in range(6):
    tx_clean[i] = A[i] * np.sin(2*np.pi*f[i]*t + ph[i])

# ground copy (ideal) but we will add noise to both in-air and ground to simulate realistic conditions
# noise levels
sig_tx = 0.05   # transmitter measurement noise (higher)
sig_gx = 0.02   # ground baseline noise (lower)

tx = tx_clean + sig_tx * np.random.randn(6, N)
gx = tx_clean + sig_gx * np.random.randn(6, N)  # ground has slightly different noise

# inject STEP anomalies (offset) at different times per channel
sidx = [6, 9, 12, 15, 18, 21]   # start indices
dur = 6                        # duration in samples
offs = [0.8, -1.0, 0.5, -0.6, 1.2, -0.9]

for i in range(6):
    si = sidx[i]; ei = min(N, si+dur)
    tx[i, si:ei] += offs[i]   # step added on transmitter side

# residual (full)
r_full = tx - gx   # ideally zero before anomaly except noise

# ----sparse sampling---- Code
# choose M sample points to send instead of all N
M = 8
# choose sampling indices - can be uniform or optimized; we'll show uniform and random choice
idx_uniform = np.linspace(0, N-1, M, dtype=int)
idx_rand = np.sort(np.random.choice(np.arange(N), size=M, replace=False))

# pick idx to use
idx = idx_uniform  # change to idx_rand to see random sampling

# sample tx and gx at those indices
tx_s = tx[:, idx]   # shape (6, M)
gx_s = gx[:, idx]
r_s = tx_s - gx_s   # sampled residuals

# ----baseline stats and detection thresholds---- Code
# compute baseline stats from first b_end samples (but only using sampled indices that fall in baseline)
b_end = 5
# find sampled indices within baseline
base_mask = idx < b_end
# baseline energy per channel using sampled points that lie in baseline window
if base_mask.sum() == 0:
    # fallback if no sampled point in baseline: use full baseline (not ideal but safe)
    mu_s = np.mean((tx[:, :b_end] - gx[:, :b_end])**2, axis=1)
    sd_s = np.std((tx[:, :b_end] - gx[:, :b_end])**2, axis=1) + 1e-12
else:
    mu_s = np.mean(r_s[:, base_mask]**2, axis=1)
    sd_s = np.std(r_s[:, base_mask]**2, axis=1) + 1e-12

# detection on sampled data: per-sample z-score then binary flag per channel
thr_z = 5.0   # z-threshold
flag_s = ( (r_s**2 - mu_s[:,None]) / sd_s[:,None] ) > thr_z   # boolean shape (6,M)

# simple pulse detection (convert sample-based flags to time intervals): for each channel, if any sampled point in anomaly region flagged -> declare anomaly
det_s = np.any(flag_s, axis=1)  # per-channel detection boolean (sparse)

# ----full detection (for comparison)---- Code
# compute per-sample z-score using full samples and baseline b_end
mu_full = np.mean(r_full[:, :b_end]**2, axis=1)
sd_full = np.std(r_full[:, :b_end]**2, axis=1) + 1e-12
flag_full = ( (r_full**2 - mu_full[:,None]) / sd_full[:,None] ) > thr_z
det_full = np.any(flag_full, axis=1)

# ----reconstruction from sparse points---- Code
# simple interpolation: linear interp from sampled points to full timeline for each channel
from numpy import interp
recon_lin = np.zeros_like(tx)
for i in range(6):
    recon_lin[i] = interp(np.arange(N), idx, tx_s[i])  # using tx sampled points to reconstruct tx
    # alternatively reconstruct delta and add to gx: delta recon
    delta_rec = interp(np.arange(N), idx, r_s[i])
    recon_lin[i] = gx[i] + delta_rec

# ----FFT diagnostics---- Code
X_full = np.abs(fft(tx, axis=1))[:, :N//2]
X_s = np.abs(fft(tx_s, axis=1))[:, :M//2]  # FFT on sampled sequence (length M)

# ----payload size estimates---- Code
# bytes if using 8-byte float for each sample value, and 1-byte per index
bytes_full = 6 * N * 8   # send all 6 channels full samples
bytes_sparse = 6 * (M * 8 + M * 1)  # value bytes + index bytes per channel (simplified)
# if we quantize values to 2 bytes (int16) after scaling, compute quantized size
bytes_sparse_q = 6 * (M * 2 + M * 1)  # quantized 2 bytes per sample + 1 byte index

# ----metrics---- Code
# detection summary table: which channels detected under full and sparse sampling
det_table = {'ch': list(range(6)), 'det_full': det_full.tolist(), 'det_sparse': det_s.tolist(), 'true_anom': [True]*6}
# compute reconstruction MSE vs true tx for each channel
mse_rec = np.mean((recon_lin - tx)**2, axis=1)

# ----plots---- Code
plt.rcParams.update({'figure.max_open_warning': 0})

# 1) Show noisy tx and gx for channel 0 as example
plt.figure(figsize=(10,4))
plt.plot(t, tx[0], label='tx noisy')
plt.plot(t, gx[0], label='gx noisy')
plt.scatter(t[idx], tx_s[0], marker='o')  # sampled points
plt.title('Example channel s0: tx vs gx with sampled points')
plt.xlabel('time (s)'); plt.ylabel('amp'); plt.legend(); plt.grid(True)
plt.show()

# 2) Residual full vs sampled for channel 0
plt.figure(figsize=(10,3))
plt.plot(t, r_full[0], label='r full')
plt.scatter(t[idx], r_s[0], label='r sampled', marker='x')
plt.title('Residual: full vs sampled (s0)'); plt.xlabel('time (s)'); plt.legend(); plt.grid(True)
plt.show()

# 3) Binary flag heatmaps for full and sparse (6x time)
plt.figure(figsize=(10,4))
plt.imshow(flag_full, aspect='auto', origin='lower', extent=[0,(N-1)/fs,0,6])
plt.title('flag_full (per-sample detection)'); plt.xlabel('time (s)'); plt.yticks(np.arange(6)+0.5, [f's{i}' for i in range(6)])
plt.colorbar()
plt.show()

plt.figure(figsize=(8,3))
plt.imshow(flag_s, aspect='auto', origin='lower', extent=[0,(M-1)/fs,0,6])
plt.title('flag_s (sampled points detection)'); plt.xlabel('sample index (of sampled vector)'); plt.yticks(np.arange(6)+0.5, [f's{i}' for i in range(6)])
plt.colorbar()
plt.show()

# 4) Reconstruction vs true for a channel
ch = 2
plt.figure(figsize=(10,3))
plt.plot(t, tx[ch], label='tx true')
plt.plot(t, recon_lin[ch], label='recon from sparse')
plt.scatter(t[idx], tx_s[ch], label='samples used', zorder=5)
plt.title(f'Channel s{ch}: reconstruction from M={M} samples'); plt.legend(); plt.grid(True)
plt.show()

# 5) Show FFT diagnostics heatmap (magnitude)
plt.figure(figsize=(10,4))
plt.imshow(X_full, aspect='auto', origin='lower', extent=[0, fs/2, 0, 6])
plt.title('FFT mag per channel (full)'); plt.xlabel('freq (Hz)'); plt.yticks(np.arange(6)+0.5, [f's{i}' for i in range(6)])
plt.colorbar(); plt.show()

# ----print outputs---- Code
print("sample indices used (idx):", idx.tolist())
print("bytes_full (send everything):", bytes_full)
print("bytes_sparse (float+index):", bytes_sparse)
print("bytes_sparse_q (int16+index):", bytes_sparse_q)
print("\ndetection table (ch, det_full, det_sparse):")
for i in range(6):
    print(i, det_full[i], det_s[i], "alarm_idx_sampled:", np.where(flag_s[i])[0].tolist())

print("\nMSE per channel for linear recon from sampled points:")
for i in range(6):
    print(f"s{i}: MSE={mse_rec[i]:.6e}")

# show payload content (sampled residuals) for channels that detected anomaly
print("\nexample payload (channel, sample_idx, value) for detected channels (sparse):")
for i in range(6):
    if det_s[i]:
        # list sampled points where flag true
        true_samples = np.where(flag_s[i])[0].tolist()
        vals = [(int(idx[j]), float(r_s[i,j])) for j in true_samples]
        print(f"ch{i}: flagged sampled pts:", vals)

# End of script

