clear; clc; close all;

%% =========================================
% PARAMETERS
% =========================================
Ts = 0.01;           % 10 ms
Fs = 1/Ts;           % 100 Hz
T  = 100;            % total time (sec)
t  = 0:Ts:T-Ts;
N  = length(t);

fault_time = 40;     % fault occurs at 40 sec
fault_idx  = fault_time/Ts;

bias = 0.3;          % acceleration bias fault

%% =========================================
% TRUE LONGITUDINAL MODEL (NO FAULT)
% =========================================
ax_true = 0.2*sin(0.2*t);
vx_true = cumtrapz(t, ax_true);
x_true  = cumtrapz(t, vx_true);

%% =========================================
% FAULTY MEASUREMENT (BIAS STEP)
% =========================================
ax_meas = ax_true;
ax_meas(fault_idx:end) = ax_meas(fault_idx:end) + bias;

vx_meas = cumtrapz(t, ax_meas);
x_meas  = cumtrapz(t, vx_meas);

%% =========================================
% FIGURE 1: RAW SIGNALS
% =========================================
figure('Name','Raw Signals');
subplot(3,1,1)
plot(t,ax_meas); xline(fault_time,'r--');
ylabel('a_x (m/s^2)'); title('Acceleration')

subplot(3,1,2)
plot(t,vx_meas); xline(fault_time,'r--');
ylabel('V_x (m/s)'); title('Velocity')

subplot(3,1,3)
plot(t,x_meas); xline(fault_time,'r--');
ylabel('x (m)'); xlabel('Time (s)'); title('Position')

%% =========================================
% ERROR SIGNAL (GROUND VS AIR IDEA)
% =========================================
dx = x_meas - x_true;

figure('Name','Position Error');
plot(t,dx,'b'); hold on;
xline(fault_time,'k--','Fault');
ylabel('Error (m)'); xlabel('Time (s)');
title('Position Difference (Should be Flat Before Fault)');
grid on;

%% =========================================
% METHOD 1: RAW DATA DETECTION
% =========================================
th1 = 0.2;
idx_m1 = find(abs(dx) > th1,1);
t_m1 = t(idx_m1);

%% =========================================
% METHOD 2: WINDOWED (SLICED) DATA
% =========================================
win = 50;                % 50 samples = 0.5 sec
numWin = floor(N/win);

win_feat = zeros(1,numWin);
win_time = zeros(1,numWin);

for k = 1:numWin
    id = (k-1)*win + 1 : k*win;
    win_feat(k) = max(abs(dx(id)));
    win_time(k) = mean(t(id));
end

idx_m2 = find(win_feat > th1,1);
t_m2 = win_time(idx_m2);

%% =========================================
% METHOD 3: MODEL PREDICTION (MIN DATA)
% =========================================
x_pred = zeros(1,N);
vx_pred = 0;

for k = 2:N
    vx_pred = vx_pred + ax_meas(k-1)*Ts;
    x_pred(k) = x_pred(k-1) + vx_pred*Ts;
end

residual = x_meas - x_pred;
th3 = 0.05;

idx_m3 = find(abs(residual) > th3,1);
t_m3 = t(idx_m3);

%% =========================================
% FIGURE 2: WINDOW OPERATION VISUAL
% =========================================
figure('Name','Windowed Detection');
stairs(win_time,win_feat,'LineWidth',1.5); hold on;
yline(th1,'r--','Threshold');
xline(fault_time,'k--','Fault');
xline(t_m2,'g','Detection');
xlabel('Time (s)');
ylabel('Window Max Error');
title('50-Sample Window Compression');
grid on;

%% =========================================
% FIGURE 3: MODEL RESIDUAL
% =========================================
figure('Name','Prediction Residual');
plot(t,residual,'m'); hold on;
yline(th3,'r--');
xline(fault_time,'k--','Fault');
xlabel('Time (s)');
ylabel('Residual (m)');
title('Model-Based Detection (1 Sample)');
grid on;

%% =========================================
% FIGURE 4: DETECTION TIME COMPARISON
% =========================================
figure('Name','Detection Delay');
bar([t_m1-fault_time, t_m2-fault_time, t_m3-fault_time])
set(gca,'XTickLabel',{'Raw','Windowed','Model'})
ylabel('Detection Delay (s)')
title('Detection Speed Comparison')
grid on;

%% =========================================
% PRINT SUMMARY
% =========================================
fprintf('\n========== SUMMARY ==========\n');
fprintf('Total samples sent (Raw)     : %d\n',N);
fprintf('Samples per window (Method2): %d\n',win);
fprintf('Samples per step (Method3)  : 1\n\n');

fprintf('Detection Times (sec)\n');
fprintf('Raw       : %.2f\n',t_m1);
fprintf('Windowed  : %.2f\n',t_m2);
fprintf('Model     : %.2f\n',t_m3);
