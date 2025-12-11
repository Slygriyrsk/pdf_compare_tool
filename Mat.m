% keypoint_anom.m
% 6 sensors, step anomalies, keypoint extraction, detection, payload
rng(0);

% ----setup----
fs = 10;           % 10 Hz (0.1s)
N = 30;
t = (0:N-1)/fs;

freqs = [0.5,1.0,1.5,0.7,1.2,0.9];
A = [1.0,0.8,0.6,1.1,0.9,0.7];

air_clean = zeros(6,N);
for ch = 1:6
    air_clean(ch,:) = A(ch) * sin(2*pi*freqs(ch) * t);
end
gx_clean = air_clean;

% ----noise----
sigma_air = 0.05;
sigma_gx  = 0.02;
air = air_clean + sigma_air * randn(6,N);
gx  = gx_clean + sigma_gx  * randn(6,N);

% ----inject STEP anomalies----
sidx = [7,10,13,16,19,22];   % MATLAB 1-based indices
dur = 6;
offs = [0.8, -1.0, 0.5, -0.6, 1.2, -0.9];
for ch = 1:6
    si = sidx(ch);
    ei = min(N, si+dur-1);
    air(ch, si:ei) = air(ch, si:ei) + offs(ch);
end

% ----keypoint extraction function (inline)----
get_keypoints = @(x) unique( [ find(diff(sign(x))~=0)+1, find(islocalmax(x)), find(islocalmin(x)) ] );

kp_idx = cell(6,1);
for ch = 1:6
    kp = get_keypoints(air(ch,:));
    if isempty(kp)
        kp = [1, N];
    end
    kp_idx{ch} = sort(kp);
end

% ----compute residual at keypoints and detect----
thr = 0.1;
flag_kp = cell(6,1);
flag_full = zeros(6,N);

payload = {};
for ch = 1:6
    idx = kp_idx{ch};
    v_air = air(ch, idx);
    v_gx  = gx(ch, idx);
    v_r   = v_air - v_gx;
    f_k   = abs(v_r) > thr;
    flag_kp{ch} = f_k;
    halfw = max(1, floor(dur/3));
    for k = 1:length(idx)
        if f_k(k)
            kpos = idx(k);
            st = max(1, kpos - halfw);
            en = min(N, kpos + halfw);
            flag_full(ch, st:en) = 1;
        end
    end
    % collect payload entries from contiguous flagged regions
    vec = flag_full(ch,:);
    in_r = false;
    for s = 1:N
        if (~in_r) && (vec(s) == 1)
            in_r = true; st = s;
        end
        if in_r && ((s == N) || (vec(s) == 1 && vec(min(s+1,N)) == 0))
            en = s; in_r = false;
            dur_samps = en - st + 1;
            off_est = mean( air(ch, st:en) - gx(ch, st:en) );
            payload{end+1} = struct('ch', ch-1, 'si', st-1, 'dur', dur_samps, 'off', off_est);
        end
    end
end

% ----reconstruct (simple interp)----
recon = gx;
for ch = 1:6
    idx = kp_idx{ch};
    xs = idx; ys = air(ch, idx);
    if length(xs) == 1
        est = ones(1,N) * ys(1);
    else
        est = interp1(xs, ys, 1:N, 'linear', 'extrap');
    end
    recon(ch,:) = est;
end

% ----FFT diagnostics----
Xair = abs(fft(air, [], 2));
Xair = Xair(:, 1:floor(N/2));

% ----plots----
figure('Position',[100 100 900 1100]);
for ch = 1:6
    subplot(6,1,ch);
    plot(t, air(ch,:), 'b', t, gx(ch,:), 'k--'); hold on;
    kx = kp_idx{ch};
    scatter((kx-1)/fs, air(ch, kx), 40, 'r', 'filled');
    xlim([0, (N-1)/fs]); ylabel(sprintf('s%d', ch-1));
    if ch==1; legend('air','gx','kp'); end
end
xlabel('time (s)');
sgtitle('Air vs Ground with keypoints (red)');

figure;
for ch = 1:6
    subplot(6,1,ch);
    stem((kp_idx{ch}-1)/fs, double(flag_kp{ch})); ylim([-0.1 1.1]);
    ylabel(sprintf('s%d', ch-1));
    if ch==1; title('binary flag at keypoints'); end
end
xlabel('time (s)');

figure;
for ch = 1:6
    subplot(6,1,ch);
    stairs((0:N-1)/fs, flag_full(ch,:)); ylim([-0.1 1.1]); ylabel(sprintf('s%d', ch-1));
    if ch==1; title('square-wave flags (expanded from kp)'); end
end
xlabel('time (s)');

figure;
imagesc([0:floor(N/2)-1], 1:6, Xair); axis xy;
xlabel('FFT bin'); ylabel('channel'); title('FFT mag (air) per channel (first N/2 bins)');
colorbar;

% ----print results----
disp('Keypoints per channel (indices, 0-based):');
for ch = 1:6
    fprintf('ch%d: [%s]\n', ch-1, num2str(kp_idx{ch}-1));
end

disp('Payload entries:');
for k = 1:length(payload)
    p = payload{k};
    fprintf('ch%d si=%d dur=%d off=%.3f\n', p.ch, p.si, p.dur, p.off);
end
