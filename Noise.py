# step_anom_demo.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
np.random.seed(2)

# params (short names)
fs = 10           # samples/sec (0.1s interval)
N = 30            # samples
t = np.arange(N)/fs

# 6 signals s0..s5 (frequencies and amps)
f = np.array([0.5, 1.0, 1.5, 0.7, 1.2, 0.9])
A = np.array([1.0, 0.8, 0.6, 1.1, 0.9, 0.7])
ph = 2*np.pi*np.random.rand(6)

# build tx (in-air) and gx (ground copy identical initially)
tx = np.zeros((6, N))
gx = np.zeros_like(tx)
for i in range(6):
    tx[i] = A[i]*np.sin(2*np.pi*f[i]*t + ph[i])
    gx[i] = tx[i].copy()

# inject STEP anomalies (offset added) at different start indices for each channel
sidx = [6, 9, 12, 15, 18, 21]  # start sample for each channel's anomaly (0-based)
dur = 6                        # duration in samples (step persists for dur samples then returns)
offs = [0.8, -1.0, 0.5, -0.6, 1.2, -0.9]  # offsets (step magnitude) for each channel

for i in range(6):
    si = sidx[i]
    ei = min(N, si+dur)
    tx[i, si:ei] += offs[i]   # add step offset during [si, ei)

# compute residual
r = tx - gx  # should be zero before anomalies

# binary flag per-sample per-channel: 1 if |residual| > thr, else 0
thr = 0.05
flag = (np.abs(r) > thr).astype(int)

# compress payload: for each channel, when flag==1 contiguous region found record (ch, start_idx, duration_samps, offset_est)
payload = []
for i in range(6):
    fvec = flag[i]
    in_region = False
    for j in range(N):
        if (not in_region) and (fvec[j]==1):
            in_region = True
            st = j
        # detect end-of-region when next sample is 0 or at last sample
        if in_region and ( (j==N-1) or (fvec[j]==1 and fvec[min(j+1,N-1)]==0) ):
            en = j
            dur_samps = en - st + 1
            off_est = float(np.mean(r[i, st:en+1]))  # simple offset estimate
            payload.append({'ch':i, 'si':st, 'dur':dur_samps, 'off':off_est})
            in_region = False

# FFTs for diagnostics
X = np.zeros((6, N), dtype=complex)
mag = np.zeros((6, N//2))
for i in range(6):
    X[i] = fft(tx[i])
    mag[i] = np.abs(X[i])[:N//2]

# PLOTS: tx vs gx, residuals + binary flags, and FFTs
plt.figure(figsize=(10,12))
for i in range(6):
    ax = plt.subplot(6,1,i+1)
    ax.plot(t, tx[i], label='tx')
    ax.plot(t, gx[i], label='gx', linewidth=0.8)
    ax.set_xlim(0, (N-1)/fs)
    ax.set_ylabel(f's{i}')
    if i==0:
        ax.legend(loc='upper right')
    si = sidx[i]; ei = min(N, si+dur)
    ax.axvspan(si/fs, ei/fs, color='0.9')
plt.xlabel('time (s)')
plt.suptitle('6 channels: tx (in-air) and gx (ground) — 30 samples @0.1s; STEP anomalies injected')
plt.tight_layout(rect=[0,0.03,1,0.95])

plt.figure(figsize=(10,12))
for i in range(6):
    ax = plt.subplot(6,1,i+1)
    ax.plot(t, r[i], label='resid')
    ax.step(t, flag[i]*np.max(np.abs(r[i]))*1.05, where='post', label='flag (scaled)', color='r')
    ax.set_xlim(0, (N-1)/fs)
    ax.set_ylabel(f'r{i}')
    if np.any(flag[i]):
        fd = np.where(flag[i]==1)[0][0]
        ax.axvline(fd/fs, color='r', linestyle='--')
    ax.legend(loc='upper right')
plt.xlabel('time (s)')
plt.suptitle('Residuals and binary flags (1 during anomaly)')
plt.tight_layout(rect=[0,0.03,1,0.95])

plt.figure(figsize=(10,8))
for i in range(6):
    ax = plt.subplot(3,2,i+1)
    ax.stem(np.arange(N//2), mag[i], use_line_collection=True)
    ax.set_xlim(0, N//2-1)
    ax.set_ylabel(f's{i} mag')
    ax.set_xlabel('bin')
plt.suptitle('FFT magnitude (first N/2 bins) for each channel (diagnostic)')
plt.tight_layout(rect=[0,0.03,1,0.95])

plt.show()

# print payload summary
print("payload (ch, start_idx, duration_samps, offset_est):")
for p in payload:
    print(p)
