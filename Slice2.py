clear; clc; close all;

%% PARAMETERS
N = 100;
t = linspace(0,2*pi,N);

A = [20 25 30 35 40 45];
anomaly_idx = [30 40 55 70 85 95];

%% GROUND SIGNALS
ground = zeros(6,N);
for i = 1:6
    ground(i,:) = A(i)*sin(t);
end

%% SHIFT TO 0–99
ground = fix(ground) + 50;
ground = min(max(ground,0),99);

%% IN-AIR SIGNALS (COPY + STEP ANOMALY)
air = ground;
for i = 1:6
    air(i,anomaly_idx(i):end) = air(i,anomaly_idx(i):end) + 20;
end
air = min(max(air,0),99);

%% PLOT RAW VS AIR
figure;
for i = 1:6
    subplot(3,2,i)
    plot(ground(i,:),'b'); hold on;
    plot(air(i,:),'r');
    xline(anomaly_idx(i),'k--');
    title(['Signal ',num2str(i)]);
    legend('Ground','Air');
end

%% DIFFERENCE (WHAT FLIGHT COMPUTER USES)
diff_sig = air - ground;

figure;
for i = 1:6
    subplot(3,2,i)
    plot(diff_sig(i,:),'k','LineWidth',1.5);
    title(['Diff Signal ',num2str(i)]);
end

%% COMPRESSION (SAFE UINT32)
C1 = zeros(1,N,'uint32');
C2 = zeros(1,N,'uint32');

for k = 1:N
    C1(k) = uint32( ...
        air(1,k) + ...
        air(2,k)*100 + ...
        air(3,k)*10000 + ...
        air(4,k)*1000000 );
    
    C2(k) = uint32( ...
        air(5,k) + ...
        air(6,k)*100 );
end

%% ANOMALY DETECTION USING COMPRESSED DIFF
dC1 = diff(double(C1));
dC2 = diff(double(C2));

figure;
plot(abs(dC1)>0,'LineWidth',2); hold on;
plot(abs(dC2)>0,'r','LineWidth',2);
title('Detected Anomaly Windows');
legend('Signals 1–4','Signals 5–6');

%% INTELLIGENT SLICE SELECTION (12 points)
energy = sum(abs(diff_sig),1);
[~,idx] = maxk(energy,8);

selected_idx = unique([1 N idx idx-1 idx+1]);
selected_idx = selected_idx(selected_idx>=1 & selected_idx<=N);

fprintf('\nSelected slices (%d total):\n',length(selected_idx));
disp(selected_idx);

%% MEMORY REPORT
bytes_raw = N * 6 * 4;
bytes_sel = length(selected_idx) * 2 * 4;

fprintf('\nMemory raw       : %d bytes\n',bytes_raw);
fprintf('Memory selected  : %d bytes\n',bytes_sel);
fprintf('Reduction factor : %.2fx\n',bytes_raw/bytes_sel);
