clear; clc; close all;

%% ===============================
% TIME SETUP
% ===============================
Fs = 100;              % Hz
Ts = 1/Fs;             % 10 ms
T  = 100;              % total time (sec)
N  = T*Fs;             % total samples
t  = (0:N-1)*Ts;

%% ===============================
% TRUE AIRCRAFT ACCELERATIONS
% ===============================
ax_true = 0.5*sin(0.2*t);     % m/s^2
az_true = 0.3*cos(0.15*t);    % m/s^2

% --- Noise (COMMENTED) ---
% ax_true = ax_true + 0.01*randn(1,N);
% az_true = az_true + 0.01*randn(1,N);

%% ===============================
% FAULT INJECTION (BIAS + STEP)
% ===============================
fault_time = 40;              % seconds
fault_idx  = fault_time*Fs;

bias = 0.3;                   % bias magnitude
step = 0.5;                   % step magnitude

ax_air = ax_true;
ax_air(fault_idx:end) = ax_air(fault_idx:end) + bias + step;

az_air = az_true;             % no fault in az

%% ===============================
% GROUND MODEL (REFERENCE)
% ===============================
Vx_g = zeros(1,N);  x_g = zeros(1,N);
Vz_g = zeros(1,N);  z_g = zeros(1,N);

for k = 2:N
    Vx_g(k) = Vx_g(k-1) + ax_true(k)*Ts;
    x_g(k)  = x_g(k-1)  + Vx_g(k)*Ts;

    Vz_g(k) = Vz_g(k-1) + az_true(k)*Ts;
    z_g(k)  = z_g(k-1)  + Vz_g(k)*Ts;
end

%% ===============================
% IN-AIR MODEL (FAULTY)
% ===============================
Vx_a = zeros(1,N);  x_a = zeros(1,N);
Vz_a = zeros(1,N);  z_a = zeros(1,N);

for k = 2:N
    Vx_a(k) = Vx_a(k-1) + ax_air(k)*Ts;
    x_a(k)  = x_a(k-1)  + Vx_a(k)*Ts;

    Vz_a(k) = Vz_a(k-1) + az_air(k)*Ts;
    z_a(k)  = z_a(k-1)  + Vz_a(k)*Ts;
end

%% ===============================
% METHOD 1: RAW DIFFERENCE
% ===============================
dx = x_a - x_g;
dVx = Vx_a - Vx_g;

figure;
subplot(2,1,1)
plot(t,dx,'LineWidth',1.5)
title('Position Difference (x)')
ylabel('Error (m)')
xline(fault_time,'r--')

subplot(2,1,2)
plot(t,dVx,'LineWidth',1.5)
title('Velocity Difference (Vx)')
ylabel('Error (m/s)')
xlabel('Time (s)')
xline(fault_time,'r--')

%% ===============================
% METHOD 2: WINDOWED (COMMENTED)
% ===============================
%{
win = 50; % 0.5 sec window
numWin = N/win;

for w = 1:numWin
    idx = (w-1)*win + 1 : w*win;
    feature(w) = max(abs(dx(idx)));
end
%}

%% ===============================
% METHOD 3: PREDICTION-BASED
% ===============================
x_pred = zeros(1,N);
Vx_pred = zeros(1,N);
residual = zeros(1,N);

for k = 2:N
    Vx_pred(k) = Vx_a(k-1) + ax_air(k)*Ts;
    x_pred(k)  = x_a(k-1)  + Vx_pred(k)*Ts;

    residual(k) = x_a(k) - x_pred(k);
end

figure;
plot(t,residual,'k','LineWidth',1.5)
title('Prediction Residual (Method 3)')
ylabel('Residual (m)')
xlabel('Time (s)')
xline(fault_time,'r--')
grid on

%% ===============================
% DATA RATE COMPARISON
% ===============================
raw_bytes = N * 6 * 4;          % rough: 6 signals
res_bytes = N * 1 * 4;          % only residual

fprintf('\nDATA COMPARISON\n');
fprintf('Raw data      : %d bytes\n',raw_bytes);
fprintf('Residual only : %d bytes\n',res_bytes);
fprintf('Reduction     : %.1fx\n',raw_bytes/res_bytes);
