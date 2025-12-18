clear; clc; close all;

%% =====================================================
% PARAMETERS
% ======================================================
Ts = 0.01;                 % Sampling time (10 ms)
Fs = 1/Ts;
T  = 100;                  % Total simulation time
t  = 0:Ts:T-Ts;
N  = length(t);

fault_time = 40;           % Fault at 40 sec
fault_idx  = fault_time/Ts;

%% =====================================================
% AIRCRAFT PITCH MODEL (CTMS)
% States: [alpha; q; theta]
% ======================================================
A = [-0.313  56.7   0;
     -0.0139 -0.426 0;
      0      56.7   0];

B = [0.232;
     0.0203;
     0];

delta_e = 0.02;            % Constant elevator input

%% =====================================================
% INITIAL CONDITIONS
% ======================================================
x_pitch = zeros(3,N);      % alpha, q, theta

%% =====================================================
% SIMULATE PITCH STATES
% ======================================================
for k = 2:N
    x_dot = A*x_pitch(:,k-1) + B*delta_e;
    x_pitch(:,k) = x_pitch(:,k-1) + Ts*x_dot;
end

alpha = x_pitch(1,:);
q     = x_pitch(2,:);
theta = x_pitch(3,:);

%% =====================================================
% TRANSLATIONAL DYNAMICS
% ======================================================
ax = 9.81 * sin(theta);      % longitudinal accel
az = 9.81 * cos(theta);      % vertical accel

Vx = cumtrapz(t, ax);
Vz = cumtrapz(t, az);

x_pos = cumtrapz(t, Vx);
z_pos = cumtrapz(t, Vz);

%% =====================================================
% INJECT STEP-BIAS FAULT (alpha sensor)
% ======================================================
alpha_fault = alpha;
alpha_fault(fault_idx:end) = alpha_fault(fault_idx:end) + 0.15;

%% =====================================================
% STACK ALL SIGNALS
% ======================================================
signals_true  = [alpha; q; theta; Vx; Vz; x_pos; z_pos];
signals_fault = [alpha_fault; q; theta; Vx; Vz; x_pos; z_pos];

names = {'\alpha','q','\theta','V_x','V_z','x','z'};

%% =====================================================
% FIGURE 1: ALL SIGNALS
% ======================================================
figure('Name','Aircraft States');
for i = 1:7
    subplot(4,2,i)
    plot(t,signals_fault(i,:),'b'); hold on;
    xline(fault_time,'r--');
    title(names{i}); grid on;
end

%% =====================================================
% DIFFERENCE (AIR vs GROUND IDEA)
% ======================================================
diff_sig = signals_fault - signals_true;

figure('Name','Difference Signals');
for i = 1:7
    subplot(4,2,i)
    plot(t,diff_sig(i,:)); hold on;
    xline(fault_time,'r--');
    title(['Diff ',names{i}]);
    grid on;
end

%% =====================================================
% METHOD 1: RAW DETECTION
% ======================================================
th = 0.05;
idx_raw = find(abs(diff_sig(1,:)) > th,1);
t_raw = t(idx_raw);

%% =====================================================
% METHOD 2: WINDOWED COMPRESSION
% ======================================================
win = 200;                 % 200 samples = 2 sec
numWin = floor(N/win);

feat = zeros(1,numWin);
t_win = zeros(1,numWin);

for k = 1:numWin
    id = (k-1)*win+1:k*win;
    feat(k) = max(abs(diff_sig(1,id)));
    t_win(k) = mean(t(id));
end

idx_win = find(feat > th,1);
t_win_det = t_win(idx_win);

figure;
stairs(t_win,feat,'LineWidth',1.5); hold on;
yline(th,'r--'); xline(fault_time,'k--');
xline(t_win_det,'g','Detected');
title('Windowed Detection');
xlabel('Time (s)');
ylabel('Window Feature');
grid on;

%% =====================================================
% METHOD 3: MODEL-BASED (MIN DATA)
% ======================================================
alpha_pred = zeros(1,N);

for k = 2:N
    alpha_pred(k) = alpha_pred(k-1) + Ts*(A(1,:)*x_pitch(:,k-1));
end

res = alpha_fault - alpha_pred;
idx_model = find(abs(res) > th,1);
t_model = t(idx_model);

figure;
plot(t,res,'m'); hold on;
yline(th,'r--');
xline(fault_time,'k--');
title('Model Residual');
grid on;

%% =====================================================
% INTEGER PACKING (32-bit CHECK)
% Pack 4 signals only (fits 32-bit)
% ======================================================
scaled = fix((signals_fault(1:4,:) + 1)*50); % 0–99

packed = zeros(1,N,'uint32');
for k = 1:N
    packed(k) = uint32( ...
        scaled(1,k) + ...
        scaled(2,k)*100 + ...
        scaled(3,k)*10000 + ...
        scaled(4,k)*1000000 );
end

max_val = max(packed);

%% =====================================================
% PERFORMANCE SUMMARY
% ======================================================
fprintf('\n========= PERFORMANCE =========\n');
fprintf('Raw detection time      : %.2f sec\n',t_raw);
fprintf('Window detection time   : %.2f sec\n',t_win_det);
fprintf('Model detection time    : %.2f sec\n',t_model);

fprintf('\n32-bit packing:\n');
fprintf('Max packed integer      : %u\n',max_val);
fprintf('32-bit limit            : %u\n',2^32-1);

fprintf('\nData Usage:\n');
fprintf('Raw samples sent        : %d\n',N*7);
fprintf('Windowed samples sent   : %d\n',numWin);
fprintf('Model samples sent      : %d\n',N);

%% =====================================================
% DETECTION DELAY COMPARISON
% ======================================================
figure;
bar([t_raw-fault_time, t_win_det-fault_time, t_model-fault_time])
set(gca,'XTickLabel',{'Raw','Window','Model'})
ylabel('Detection Delay (sec)')
title('Detection Delay Comparison')
grid on;
