# flight_sig_demo.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
np.random.seed(1)

# sampling
fs = 10       # 10 samples/sec -> 0.1s interval
N = 30        # 30 samples
t = np.arange(N) / fs

# 6 signals (s0..s5)
f = np.array([0.5, 1.0, 1.5, 0.7, 1.2, 0.9])  # Hz
A = np.array([1.0, 0.8, 0.6, 1.1, 0.9, 0.7])
ph = 2*np.pi*np.random.rand(6)

# build tx and gx (identical initially)
tx = np.zeros((6, N))
gx = np.zeros_like(tx)
for i in range(6):
    tx[i] = A[i] * np.sin(2*np.pi*f[i]*t + ph[i])
    gx[i] = tx[i].copy()

# anomaly injection: different start idx for each channel
sidx = [6, 9, 12, 15, 18, 21]   # starting sample index
dur = 6                         # duration in samples (~0.6s)
extra_f = [3.0, 2.5, 4.0, 3.5, 2.0, 5.0]

for i in range(6):
    si = sidx[i]
    ei = min(N, si+dur)
    tx[i, si:ei] += 0.8 * np.sin(2*np.pi*extra_f[i] * t[si:ei] + 0.2)
    tx[i, si:ei] *= 1.3    # amplitude change as well

# residual
r = tx - gx   # should be all zeros before anomalies

# energy per-sample and baseline stats (first 5 samples)
e = r**2
b_end = 5
mu_e = e[:, :b_end].mean(axis=1)
sd_e = e[:, :b_end].std(axis=1) + 1e-12
ez = (e - mu_e[:,None]) / sd_e[:,None]

# CUSUM per channel on ez (one-sided)
k = 1.0
cus = np.zeros_like(ez)
for i in range(6):
    g = 0.0
    for j in range(N):
        g = max(0.0, g + (ez[i,j] - k))
        cus[i,j] = g

# detection thresholds
alpha = 3.0
h = 5.0
alarm = np.full(6, -1, dtype=int)
for i in range(6):
    for j in range(N):
        if (ez[i,j] > alpha) or (cus[i,j] > h):
            alarm[i] = j
            break

# FFTs of full tx signals (for visualization)
X = np.zeros((6, N), dtype=complex)
mag = np.zeros((6, N//2))
for i in range(6):
    X[i] = fft(tx[i])
    mag[i] = np.abs(X[i])[:N//2]

# Payload: top-K FFT bins of delta from alarm index for dur samples
K = 3
payload = {}
recon = np.zeros_like(tx)
for i in range(6):
    if alarm[i] >= 0:
        si = alarm[i]
        ei = min(N, si+dur)
        d = r[i, si:ei]
        M = len(d)
        if M == 0:
            payload[i] = None
            recon[i] = gx[i].copy()
            continue
        D = fft(d, n=M)
        idx = np.argsort(np.abs(D)[:M//2])[::-1][:K]
        payload[i] = {'si':si, 'M':M, 'bins': idx.tolist(), 'coeff': [complex(c) for c in D[idx]]}
        # reconstruct from bins
        Xr = np.zeros(M, dtype=complex)
        for b, c in zip(idx, D[idx]):
            Xr[b] = c
            if b != 0 and b < M//2:
                Xr[-b] = np.conj(c)
        xr = np.real(ifft(Xr))
        recon[i] = gx[i].copy()
        recon[i, si:ei] += xr
    else:
        recon[i] = gx[i].copy()

# PLOTS
# 1) six channels time domain tx vs gx (30 samples)
plt.figure(figsize=(10,8))
for i in range(6):
    ax = plt.subplot(6,1,i+1)
    ax.plot(t, tx[i], label='tx')
    ax.plot(t, gx[i], label='gx', linewidth=0.8)
    ax.set_xlim(0, (N-1)/fs); ax.set_ylabel(f's{i}')
    if alarm[i] >= 0:
        si = alarm[i]; ei = min(N, si+dur)
        ax.axvspan(si/fs, ei/fs, color='0.9'); ax.axvline(si/fs, color='r', linestyle='--')
    if i == 0:
        ax.legend()
plt.xlabel('time (s)')
plt.suptitle('6 channels: tx (in-air) and gx (ground) — 30 samples @0.1s')
plt.tight_layout(rect=[0,0.03,1,0.95])

# 2) for each channel: time snippet and FFT mag
plt.figure(figsize=(12,10))
for i in range(6):
    ax1 = plt.subplot(6,2,2*i+1)
    ax1.plot(t, tx[i], label='tx'); ax1.plot(t, gx[i], label='gx', linewidth=0.8)
    ax1.set_xlim(0,(N-1)/fs)
    if alarm[i] >=0:
        ax1.axvspan(alarm[i]/fs, min(N,alarm[i]+dur)/fs, color='0.9')
    if i==0: ax1.legend()
    ax2 = plt.subplot(6,2,2*i+2)
    ax2.stem(np.arange(N//2), mag[i], markerfmt=' ', use_line_collection=True)
    ax2.set_xlim(0, N//2-1)
    ax2.set_xlabel('bin')
plt.suptitle('Left: time signals, Right: FFT mag (first N/2 bins)')
plt.tight_layout(rect=[0,0.03,1,0.95])

# 3) residuals and detection markers
plt.figure(figsize=(10,6))
for i in range(6):
    ax = plt.subplot(3,2,i+1)
    ax.plot(t, r[i], label='r')
    ax.set_xlim(0,(N-1)/fs); ax.set_ylabel(f'r{i}')
    if alarm[i] >= 0:
        ax.axvspan(alarm[i]/fs, min(N,alarm[i]+dur)/fs, color='0.9')
        ax.axvline(alarm[i]/fs, color='r', linestyle='--', label='det')
    ax.legend()
plt.suptitle('Residuals per channel and detection markers')
plt.tight_layout(rect=[0,0.03,1,0.95])

# 4) CUSUM heatmap
plt.figure(figsize=(8,4))
plt.imshow(cus, aspect='auto', origin='lower', extent=[0,(N-1)/fs,0,6])
plt.colorbar(label='CUSUM')
plt.yticks(np.arange(6)+0.5, [f's{i}' for i in range(6)])
plt.xlabel('time (s)'); plt.title('CUSUM per channel')

plt.show()

# print alarms + payload summary
print("alarms (first alarm sample index per channel):", alarm.tolist())
print("payload (top-K bins) for channels with alarms:")
for i in sorted(payload.keys()):
    p = payload[i]
    if p is None:
        print(f"ch{i}: no payload (M=0)")
    else:
        print(f"ch{i}: si={p['si']}, M={p['M']}, bins={p['bins']}, coeff_count={len(p['coeff'])}")
