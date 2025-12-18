clear; clc; close all;

%% =====================================================
% GLOBAL PARAMETERS
%% =====================================================
Fs = 100;                 % Hz
Ts = 1/Fs;
T  = 100;                % seconds
t  = 0:Ts:T-Ts;
N  = length(t);

%% =====================================================
% SIGNAL DEFINITIONS (LONGITUDINAL)
%% =====================================================
names = {'\alpha','\theta','q','V_x','V_z','x','z','\delta_e'};
freq  = [0.2 0.15 0.3 0.1 0.12 0.05 0.07 0.02];
amp   = [5   3    2   20   15   100  80   1];

ns = length(names);
signals_true = zeros(ns,N);

for i = 1:ns
    signals_true(i,:) = amp(i)*sin(2*pi*freq(i)*t);
end

%% =====================================================
% FAULT INJECTION (STEP BIAS)
%% =====================================================
fault_times = [20 30 40 50 60 70 80 90];
fault_idx   = round(fault_times * Fs);
bias        = [1 0.8 0.5 3 2 10 8 0.2];

signals_fault = signals_true;

for i = 1:ns
    signals_fault(i,fault_idx(i):end) = ...
        signals_fault(i,fault_idx(i):end) + bias(i);
end

%% =====================================================
% FIGURE 1: SIGNALS WITH FAULT
%% =====================================================
figure('Name','Aircraft Signals with Fault');
for i = 1:ns
    subplot(4,2,i)
    plot(t,signals_fault(i,:),'b'); hold on;
    xline(fault_times(i),'r--','LineWidth',1.2);
    title(names{i}); grid on;
end

%% =====================================================
% DIFFERENCE SIGNALS
%% =====================================================
%diff_sig = abs(signals_fault - signals_true);
diff_sig = [zeros(8,1) diff(signals_fault,1,2)];


figure('Name','Difference Signals');
for i = 1:ns
    subplot(4,2,i)
    plot(t,diff_sig(i,:),'k'); hold on;
    xline(fault_times(i),'r--');
    title(['Diff ',names{i}]); grid on;
end

%% =====================================================
% METHOD 1: RAW DETECTION
%% =====================================================
th = 0.3;   % smaller threshold for impulse
det_raw   = NaN(1,8);
delay_raw = NaN(1,8);

for i = 1:8
    idx = find(abs(diff_sig(i,:)) > th, 1, 'first');

    if ~isempty(idx)
        det_raw(i)   = t(idx);
        delay_raw(i) = det_raw(i) - fault_times(i);
    end
end

%% =====================================================
% METHOD 2: WINDOW (SLICED) DETECTION
%% =====================================================
win = 100;                % 1 second
numWin = floor(N/win);
feat  = zeros(8,numWin);
t_win = zeros(1,numWin);

for w = 1:numWin
    idx = (w-1)*win + 1 : w*win;
    t_win(w) = mean(t(idx));
    feat(:,w) = mean(abs(diff_sig(:,idx)),2);
end

det_win   = NaN(1,8);
delay_win = NaN(1,8);

for i = 1:8
    idx = find(feat(i,:) > th & t_win >= fault_times(i),1);

    if ~isempty(idx)
        det_win(i)   = t_win(idx);
        delay_win(i) = det_win(i) - fault_times(i);
    end
end

%% WINDOW FEATURE PLOTS
figure('Name','Window Features');
for i = 1:ns
    subplot(4,2,i)
    stairs(t_win,feat(i,:),'b','LineWidth',1.2); hold on;
    yline(th,'r--');
    xline(fault_times(i),'k:');
    title(['Window Feature ',names{i}]); grid on;
end

%% =====================================================
% METHOD 3: MODEL / RESIDUAL DETECTION
%% =====================================================
det_est   = NaN(1,ns);
delay_est = NaN(1,ns);

for i = 1:ns
    res = [0 diff(signals_fault(i,:) - signals_true(i,:))];
    idx = find(abs(res) > th,1);

    if ~isempty(idx)
        det_est(i)   = t(idx);
        delay_est(i) = det_est(i) - fault_times(i);
    end

    figure;
    plot(t,res,'k'); hold on;
    yline(th,'r--');
    xline(fault_times(i),'b--');
    title(['Residual ',names{i}]); grid on;
end

%% =====================================================
% ERROR PERCENTAGE
%% =====================================================
err_pct = zeros(3,ns);
for i = 1:ns
    err_pct(1,i) = abs(delay_raw(i))/fault_times(i)*100;
    err_pct(2,i) = abs(delay_win(i))/fault_times(i)*100;
    err_pct(3,i) = abs(delay_est(i))/fault_times(i)*100;
end

%% =====================================================
% 32-BIT PACKING (4 SIGNALS)
%% =====================================================
sig4 = signals_fault(1:4,:);
mx = max(abs(sig4),[],2);

scaled = fix((sig4 + mx)./(2*mx) * 99);
packed = uint32(zeros(1,N));

for k = 1:N
    packed(k) = uint32( ...
        scaled(1,k) + ...
        scaled(2,k)*100 + ...
        scaled(3,k)*10000 + ...
        scaled(4,k)*1000000 );
end

%% =====================================================
% PERFORMANCE COMPARISON
%% =====================================================
figure;
bar([mean(delay_raw,'omitnan') mean(delay_win, 'omitnan') mean(delay_est, 'omitnan')]);
set(gca,'XTickLabel',{'Raw','Window','Estimation'});
ylabel('Avg Detection Delay (s)');
title('Detection Delay Comparison'); grid on;

figure;
bar(mean(err_pct,2));
set(gca,'XTickLabel',{'Raw','Window','Estimation'});
ylabel('% Error');
title('Detection Error Comparison'); grid on;

%% =====================================================
% SUMMARY PRINT
%% =====================================================
fprintf('\n===== SUMMARY =====\n');
fprintf('Sampling: %d Hz | Duration: %d sec\n',Fs,T);
fprintf('Avg Delay Raw        : %.3f s\n',mean(delay_raw,'omitnan'));
fprintf('Avg Delay Window     : %.3f s\n',mean(delay_win, 'omitnan'));
fprintf('Avg Delay Estimation : %.3f s\n',mean(delay_est, 'omitnan'));
fprintf('Max Packed Value     : %u\n',max(packed));
fprintf('32-bit Limit         : %u\n',2^32-1);
