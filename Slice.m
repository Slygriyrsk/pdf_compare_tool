clear; clc; close all;

%% ---------------------------------------
% PARAMETERS
% ----------------------------------------
N = 100;              % total slices
t = linspace(0,2*pi,N);

% Different amplitudes for 6 signals
A = [20 25 30 35 40 45];    

% Different anomaly positions
anomaly_idx = [30 40 55 70 85 95];      

%% ---------------------------------------
% GENERATE 6 SINE SIGNALS
% ----------------------------------------
signals = zeros(6,N);
for i = 1:6
    signals(i,:) = A(i) * sin(t);
end

figure; 
for i = 1:6
    subplot(3,2,i)
    plot(signals(i,:)); title(['Signal ',num2str(i),' (Raw)']);
    ylabel('Amplitude');
end

%% ---------------------------------------
% SHIFT SIGNALS TO 0–99 RANGE
% b = fix(a) + 50
% ----------------------------------------
shifted = fix(signals) + 50;

figure;
for i = 1:6
    subplot(3,2,i)
    plot(shifted(i,:)); title(['Signal ',num2str(i),' Shifted 0–99']);
end

%% ---------------------------------------
% ADD STEP-JUMP ANOMALY TO EACH SIGNAL
% ----------------------------------------
anomaly_signals = shifted;
for i = 1:6
    idx = anomaly_idx(i);
    anomaly_signals(i,idx:end) = anomaly_signals(i,idx:end) + 20;
end

figure;
for i = 1:6
    subplot(3,2,i)
    plot(anomaly_signals(i,:)); hold on;
    xline(anomaly_idx(i),'r--','LineWidth',1.5);
    title(['Signal ',num2str(i),' With Step-Jump']);
end

%% ---------------------------------------
% COMPRESS EACH SLICE:
% FORMAT: s1 + s2*100 + s3*10000 + ... 
% Each component is 2 digits (00–99)
% ----------------------------------------
compressed = zeros(1,N);

for k = 1:N
    c = 0;
    mul = 1;
    for s = 1:6
        c = c + anomaly_signals(s,k) * mul;
        mul = mul * 100;
    end
    compressed(k) = c;
end

%% PLOT COMPRESSED VALUES
figure; 
plot(compressed,'LineWidth',1.2);
title('Compressed Value per Slice');
xlabel('Slice');
ylabel('Compressed Integer');

%% ---------------------------------------
% MINIMUM SLICE SELECTION
% Use 12 optimal KP points (cycle-mid + peaks)
% ----------------------------------------
selected_idx = round(linspace(1,N,12));   % 12 equally-paced keypoints

selected_values = anomaly_signals(:,selected_idx);

%% ---------------------------------------
% RECONSTRUCT USING CUBIC INTERPOLATION
% ----------------------------------------
recon = zeros(6,N);
for i = 1:6
    recon(i,:) = interp1(selected_idx,selected_values(i,:),1:N,'pchip');
end

%% PLOT ORIGINAL VS RECONSTRUCTION
figure;
for i = 1:6
    subplot(3,2,i)
    plot(anomaly_signals(i,:),'b','LineWidth',1); hold on;
    plot(recon(i,:),'r--','LineWidth',1.2);
    title(['Signal ',num2str(i),' Original vs Recon (12 points)']);
    legend('Original','Reconstructed');
end

%% ---------------------------------------
% PRINT MEMORY USAGE
% ----------------------------------------
bytes_raw  = N * 6 * 4;         % 4 bytes per 32-bit value
bytes_12   = 12 * 6 * 4;        % only 12 slices
compression_factor = bytes_raw / bytes_12;

fprintf("\n========== MEMORY REPORT ==========\n");
fprintf("Raw signal size      : %d bytes\n", bytes_raw);
fprintf("Compressed (12 pts)  : %d bytes\n", bytes_12);
fprintf("Compression Factor   : %.2f X\n", compression_factor);

%% PRINT BIT REPRESENTATION VISUAL
fprintf("\nBit format (one example point):\n");

example_point = anomaly_signals(:,40);
fprintf("Raw 32-bit values:\n");
disp(example_point);

fprintf("Binary (LSB to MSB order):\n");
for s = 1:6
    fprintf("S%d : %s\n",s, dec2bin(example_point(s),32));
end

fprintf("\nCompressed integer at slice 40: %d\n",compressed(40));

l
