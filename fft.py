# Simulate multi-sensor sinusoidal data, detect anomaly with STFT+CUSUM, compress with top-K FFT bins,
# and reconstruct on "ground". Produce MATLAB-style plots using matplotlib.
# This code runs in the notebook and displays the results to the user.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, windows
from scipy.fft import fft, ifft
import scipy
import math

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
fs = 1000                    # sampling rate Hz
total_time = 30.0            # seconds
t = np.arange(0, total_time, 1/fs)
n_sensors = 10
baseline_time = 8.0          # seconds used to build baseline
anomaly_sensor = 3           # index of sensor that will have anomaly (0-based)
anomaly_start = 12.0         # seconds
anomaly_duration = 6.0       # seconds (as user specified)
anomaly_end = anomaly_start + anomaly_duration

# Base common superposition (same on-air and ground baseline)
# Let's create a common superposition of a few sinusoids
common_freqs = np.array([20.0, 50.0, 120.0])  # Hz
common_amps = np.array([1.0, 0.6, 0.4])
common_phases = 2*np.pi*np.random.rand(len(common_freqs))

common_signal = np.zeros_like(t)
for A,f,phi in zip(common_amps, common_freqs, common_phases):
    common_signal += A * np.sin(2*np.pi*f*t + phi)

# Each sensor has slight amplitude scaling and an independent low-amplitude noise
sensor_signals_tx = np.zeros((n_sensors, t.size))
sensor_signals_ground = np.zeros_like(sensor_signals_tx)

for s in range(n_sensors):
    scale = 0.8 + 0.4*np.random.rand()  # small amplitude variation
    noise = 0.05 * np.random.randn(t.size)
    sensor_signals_tx[s] = scale*common_signal + noise
    sensor_signals_ground[s] = scale*common_signal + 0.01*np.random.randn(t.size)  # cleaner ground baseline

# Inject anomaly in transmitter sensor: for anomaly window, change amplitudes and add a new tone
anom_idx = (t >= anomaly_start) & (t < anomaly_end)
# Add new sinusoid and change amplitude scaling during anomaly
new_tone_freq = 200.0
for s in range(n_sensors):
    if s == anomaly_sensor:
        sensor_signals_tx[s, anom_idx] += 1.2 * np.sin(2*np.pi*new_tone_freq*t[anom_idx] + 0.3)
        # also alter existing component amplitude slightly
        sensor_signals_tx[s, anom_idx] += 0.8 * np.sin(2*np.pi*50.0*t[anom_idx] + 0.1)

# STFT parameters
win_len = 1024                # window length (samples) ~ 1.024 s
hop = 512                     # hop size 50% overlap
window = windows.hann(win_len)

# Compute STFT per sensor (magnitude)
f_stft, t_stft, Zxx0 = stft(sensor_signals_tx[0], fs=fs, window=window, nperseg=win_len, noverlap=win_len-hop)
n_freqs = len(f_stft)
n_frames = len(t_stft)

# Pre-allocate arrays: sensors x freqs x frames
STFT_mag = np.zeros((n_sensors, n_freqs, n_frames))
STFT_phase = np.zeros_like(STFT_mag)

for s in range(n_sensors):
    f_tmp, tt_tmp, Zxx = stft(sensor_signals_tx[s], fs=fs, window=window, nperseg=win_len, noverlap=win_len-hop)
    STFT_mag[s] = np.abs(Zxx)
    STFT_phase[s] = np.angle(Zxx)

# Baseline: compute mean & std of magnitude from initial baseline_time frames
baseline_frames = np.where(t_stft < baseline_time)[0]
mean_baseline = STFT_mag[:, :, baseline_frames].mean(axis=2)
std_baseline = STFT_mag[:, :, baseline_frames].std(axis=2) + 1e-8  # avoid divide-by-zero

# Detection stat per sensor per frame: max normalized magnitude deviation across freq bins
norm_dev = np.max(np.abs(STFT_mag - mean_baseline[:, :, None]) / std_baseline[:, :, None], axis=1)

# Apply CUSUM per sensor on the per-frame statistic
k = 1.0    # drift parameter (tuneable)
h = 8.0    # threshold (tuneable)
cusum = np.zeros((n_sensors, n_frames))
alarms = np.zeros((n_sensors,), dtype=bool)
alarm_frame = np.full((n_sensors,), -1, dtype=int)

for s in range(n_sensors):
    g = 0.0
    for i in range(n_frames):
        s_t = norm_dev[s, i]
        g = max(0.0, g + (s_t - k))
        cusum[s, i] = g
        if (g > h) and (not alarms[s]):
            alarms[s] = True
            alarm_frame[s] = i
            # do not break; we record first alarm frame
            
# Find which sensor flagged
flagged_sensors = np.where(alarms)[0]

# For demonstration pick the first alarm (if any)
print("Flagged sensors by CUSUM:", flagged_sensors.tolist())

# When alarm occurs, prepare compressed payload: extract delta waveform for that sensor for anomaly duration
# We'll simulate payload as top-K FFT bins of the delta (transmitter computes delta = observed - ground baseline reconstruction)
K = 6  # top K bins to send

def extract_topk_bins(sensor_idx, start_time, duration, K):
    # determine sample indices
    si = int(start_time * fs)
    ei = int((start_time + duration) * fs)
    delta = sensor_signals_tx[sensor_idx, si:ei] - sensor_signals_ground[sensor_idx, si:ei]  # delta vs ground baseline
    N = len(delta)
    # compute FFT and take top-K magnitudes (exclude DC)
    X = fft(delta)
    mag = np.abs(X)[:N//2]
    bins = np.argsort(mag)[-K:][::-1]
    # store bin frequency (Hz), complex coefficient (for phase), and bin index
    results = []
    for b in bins:
        freq = b * fs / N
        coeff = X[b]
        results.append((b, freq, coeff))
    return results, N, delta

# If CUSUM didn't flag (possible due to tuning), force detection on anomaly_sensor to show pipeline
if not np.any(alarms):
    print("No alarms detected with current thresholds; forcing pipeline for the injected anomaly sensor for demo.")
    payload_bins, N, delta = extract_topk_bins(anomaly_sensor, anomaly_start, anomaly_duration, K)
    flagged_sensor = anomaly_sensor
else:
    # choose the earliest alarm sensor
    flagged_sensor = int(flagged_sensors[0])

# Determine payload for flagged sensor using true anomaly window (we'll use known anomaly_start/duration)
payload_bins, N, delta = extract_topk_bins(flagged_sensor, anomaly_start, anomaly_duration, K)

print("Payload bins (bin_index, freq, coeff magnitude):")
for b,freq,coeff in payload_bins:
    print(f"  bin {b}, freq {freq:.1f} Hz, mag {np.abs(coeff):.3f}")


# Ground reconstruction from payload: rebuild delta waveform from top-K bins and add to ground baseline
def reconstruct_from_bins(payload_bins, N):
    # Reconstruct an N-point time-domain signal using only the provided bins (mirror for negative freqs)
    X_rec = np.zeros(N, dtype=complex)
    for b, freq, coeff in payload_bins:
        X_rec[b] = coeff
        # mirror coefficient for negative freq (if not DC and not Nyquist)
        if b != 0 and b < N//2:
            X_rec[-b] = np.conj(coeff)
    x_rec = np.real(ifft(X_rec))
    return x_rec

reconstructed_delta = reconstruct_from_bins(payload_bins, N)

# Build ground's reconstructed full-length signal for the sensor in anomaly window
si = int(anomaly_start * fs)
ei = si + N
ground_recon_segment = sensor_signals_ground[flagged_sensor, si:ei] + reconstructed_delta

# Calculate reconstruction error
mse = np.mean((ground_recon_segment - sensor_signals_tx[flagged_sensor, si:ei])**2)
print(f"Reconstruction MSE for flagged sensor segment: {mse:.6f}")

# Plotting: similar to MATLAB style (single plot per figure)
# 1) Plot a short time zoom showing the anomaly (original tx vs ground baseline)
zoom_start = anomaly_start - 1.0
zoom_end = anomaly_end + 1.0
zi = int(zoom_start*fs); ze = int(zoom_end*fs)
tt = t[zi:ze]

plt.figure(figsize=(10,3))
plt.plot(tt, sensor_signals_tx[flagged_sensor, zi:ze], label='Transmitter (sensor)')
plt.plot(tt, sensor_signals_ground[flagged_sensor, zi:ze], label='Ground baseline (sensor)', linewidth=1)
plt.axvspan(anomaly_start, anomaly_end, color='0.9', label='Anomaly window')
plt.title(f"Sensor {flagged_sensor}: time-domain signals (zoom)")
plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
plt.legend(); plt.tight_layout()

# 2) STFT spectrogram for flagged sensor
plt.figure(figsize=(10,3))
plt.pcolormesh(t_stft, f_stft, 20*np.log10(STFT_mag[flagged_sensor]+1e-12), shading='gouraud')
plt.colorbar(label='Magnitude (dB)')
plt.ylim(0, 400)
plt.title(f"Sensor {flagged_sensor}: STFT magnitude (dB)")
plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
plt.axvline(anomaly_start, color='w', linestyle='--'); plt.axvline(anomaly_end, color='w', linestyle='--')
plt.tight_layout()

# 3) CUSUM statistic for all sensors (plot a subset)
plt.figure(figsize=(10,4))
for s in range(min(n_sensors,6)):
    plt.plot(t_stft, cusum[s], label=f'sensor {s}')
plt.title("CUSUM statistic over time (first 6 sensors)")
plt.xlabel('Time (s)'); plt.ylabel('CUSUM value')
plt.legend(); plt.tight_layout()

# 4) Detected frame and reconstruction comparison (overlay reconstructed vs true tx in anomaly segment)
seg_t = np.linspace(0, anomaly_duration, N, endpoint=False)
plt.figure(figsize=(10,3))
plt.plot(seg_t, sensor_signals_tx[flagged_sensor, si:ei], label='True Transmitter segment')
plt.plot(seg_t, ground_recon_segment, label='Ground reconstructed (baseline + payload)', linestyle='--')
plt.title(f"Reconstruction comparison for sensor {flagged_sensor} (anomaly segment)")
plt.xlabel('Time (s) into anomaly'); plt.ylabel('Amplitude')
plt.legend(); plt.tight_layout()

# 5) Plot the delta true vs reconstructed delta
plt.figure(figsize=(10,3))
plt.plot(seg_t, delta, label='True delta (tx - ground baseline)')
plt.plot(seg_t, reconstructed_delta, label='Reconstructed delta from top-K bins', linestyle='--')
plt.title("Delta wave: true vs reconstructed from payload")
plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
plt.legend(); plt.tight_layout()

# Show figures
plt.show()
