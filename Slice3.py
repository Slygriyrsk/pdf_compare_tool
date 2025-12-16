clear; clc; close all;

%% PARAMETERS
N = 100;                % total samples
Fs = 100;
t = linspace(0,1,N);

winSize = 20;
numWin = N / winSize;

A = [20 25 30 35 40 45];          % amplitudes
an_idx = [30 40 55 70 85 95];     % anomaly points

%% GROUND SIGNALS
g = zeros(6,N);
for i = 1:6
    g(i,:) = A(i)*sin(2*pi*t);
end

%% SHIFT TO 0–99
g = fix(g) + 50;
g = min(max(g,0),99);

%% AIR SIGNALS + STEP ANOMALY
a = g;
for i = 1:6
    a(i,an_idx(i):end) = a(i,an_idx(i):end) + 20;
end
a = min(max(a,0),99);

%% PLOT SIGNALS WITH WINDOWS
figure;
for i = 1:6
    subplot(3,2,i)
    plot(g(i,:),'b'); hold on;
    plot(a(i,:),'r');
    for w = 1:numWin
        xline(w*winSize,'k:');
    end
    title(['Signal ',num2str(i)]);
    legend('Ground','Air');
end

%% WINDOW-BASED COMPRESSION
W1_g = zeros(1,numWin,'uint32');
W2_g = zeros(1,numWin,'uint32');
W1_a = zeros(1,numWin,'uint32');
W2_a = zeros(1,numWin,'uint32');

fprintf('\nWINDOW NUMBERS (FORMAT DEMO)\n');
fprintf('---------------------------------\n');

for w = 1:numWin
    idx = (w-1)*winSize + 1 : w*winSize;

    gmax = max(g(:,idx),[],2);
    amax = max(a(:,idx),[],2);

    % pack signals 1–4
    W1_g(w) = uint32(gmax(1) + gmax(2)*100 + gmax(3)*10000 + gmax(4)*1000000);
    W1_a(w) = uint32(amax(1) + amax(2)*100 + amax(3)*10000 + amax(4)*1000000);

    % pack signals 5–6
    W2_g(w) = uint32(gmax(5) + gmax(6)*100);
    W2_a(w) = uint32(amax(5) + amax(6)*100);

    fprintf('Window %d:\n',w);
    fprintf(' W1 (s1–s4) Ground=%u  Air=%u\n',W1_g(w),W1_a(w));
    fprintf(' W2 (s5–s6) Ground=%u  Air=%u\n\n',W2_g(w),W2_a(w));
end

%% FAST ANOMALY DETECTION
d1 = abs(double(W1_a - W1_g)) > 0;
d2 = abs(double(W2_a - W2_g)) > 0;

figure;
stem(d1,'LineWidth',2); hold on;
stem(d2,'r','LineWidth',2);
title('Anomaly Detection per Window');
xlabel('Window');
ylabel('Anomaly Flag');
legend('Signals 1–4','Signals 5–6');

%% FFT (FOR UNDERSTANDING ONLY)
figure;
fft_mag = abs(fft(g(1,:)));
plot(fft_mag);
title('FFT of Signal 1 (Ground)');
xlabel('Frequency Bin');
ylabel('|FFT|');

%% MEMORY COMPARISON
bytes_raw = N * 6 * 4;
bytes_comp = numWin * 2 * 4;

fprintf('Raw data size      : %d bytes\n',bytes_raw);
fprintf('Compressed size    : %d bytes\n',bytes_comp);
fprintf('Reduction factor   : %.2fx\n',bytes_raw/bytes_comp);
