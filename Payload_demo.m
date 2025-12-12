% pack_payload_demo.m
% MATLAB version of the Python demo. Use MATLAB R2019+.

rng(0);
fs = 50; T = 30; N = fs*T; t = (0:N-1)/fs;
names = {'roll','pitch','yaw','roll_rate','pitch_rate','yaw_rate'};
freqs = [0.5 0.7 1.1 0.6 0.9 1.3];
amps  = [1.0 1.2 0.9 1.1 0.95 0.8];

air_clean = zeros(6,N);
for ch=1:6
    air_clean(ch,:) = amps(ch)*sin(2*pi*freqs(ch)*t);
end
gx_clean = air_clean;

sig_air = 0.02; sig_gx = 0.008;
air = air_clean + sig_air*randn(size(air_clean));
gx  = gx_clean + sig_gx*randn(size(gx_clean));

anoms = [1 10.0 0.6 1.5; 2 15.0 0.8 -1.2; 3 20.0 1.0 2.0; 4 8.5 0.4 0.9; 5 12.0 0.7 -1.6; 6 22.0 1.2 1.2];
air2 = air;
for k=1:size(anoms,1)
    ch = anoms(k,1); s0 = anoms(k,2); L = anoms(k,3); val = anoms(k,4);
    si = round(s0*fs); ei = min(N, round((s0+L)*fs));
    air2(ch, si:ei) = air2(ch, si:ei) + val;
end
res_full = air2 - gx;

% keypoints: smoothing + zc + peaks/troughs
kp_idx_g = cell(6,1);
for ch=1:6
    xs = sgolayfilt(gx(ch,:),3,101);
    zc = find(xs(1:end-1)<0 & xs(2:end)>=0) + 1;
    [pk,~] = findpeaks(xs);
    [tr,~] = findpeaks(-xs);
    idx = unique([zc, pk', tr']);
    kp_idx_g{ch} = sort(idx);
end

% print counts
disp('Keypoint counts:');
for ch=1:6
    fprintf('%s: %d pts (%0.1f pts/s)\n', names{ch}, numel(kp_idx_g{ch}), numel(kp_idx_g{ch})/T);
end

% detect flagged kp using MAD threshold on diff
flag_kp = cell(6,1); events = [];
for ch=1:6
    idx = kp_idx_g{ch};
    v_g = gx(ch, idx); v_a = air2(ch, idx);
    diff = v_a - v_g;
    base_mask = idx < 5*fs;
    if sum(base_mask)>0
        med = median(diff(base_mask)); mad = median(abs(diff(base_mask)-med));
    else
        med = median(diff); mad = median(abs(diff-med));
    end
    thr = med + 6*mad;
    f = double(abs(diff) > thr);
    flag_kp{ch} = f;
    for kpos_i = 1:sum(f)
        % pack event: store ch, idx(kpos_i), diff ...
        % (we will pack later)
    end
end

% collect events
evcnt = 0;
for ch=1:6
    idx = kp_idx_g{ch};
    f = flag_kp{ch};
    for i=1:numel(idx)
        if f(i)
            evcnt = evcnt + 1;
            events(evcnt).ch = ch-1;
            events(evcnt).kpos = idx(i);
            events(evcnt).amp_d = air2(ch, idx(i)) - gx(ch, idx(i));
        end
    end
end
fprintf('Total flagged events: %d\n', evcnt);

% quantize amp_d to 6-bit signed [-32..31] scaling by max observed
if evcnt==0
    disp('No events');
else
    maxabs = max(abs([events.amp_d]));
    if maxabs<=0, scale_amp=1; else scale_amp = 31/maxabs; end
    % pack into uint32 array
    packed = zeros(evcnt,1,'uint32');
    tic;
    for i=1:evcnt
        ch = bitand(events(i).ch,7);
        idx = bitand(events(i).kpos,65535);
        q = round(events(i).amp_d*scale_amp);
        q = max(-32,min(31,q));
        if q<0, q_u = q + 64; else q_u = q; end
        off_u = 0;
        flag = 1;
        word = uint32(bitshift(uint32(ch),29)) + uint32(bitshift(uint32(idx),13)) + uint32(bitshift(uint32(q_u),7)) + uint32(bitshift(uint32(off_u),1)) + uint32(flag);
        packed(i) = word;
    end
    pack_time = toc;
    packed_bytes = numel(packed)*4;
    fprintf('Packed %d events into %d bytes. pack_time = %.2f ms\n', evcnt, packed_bytes, pack_time*1000);
end

% raw bytes: send raw anomaly windows per channel
raw_bytes = 0;
for k=1:size(anoms,1)
    si = round(anoms(k,2)*fs); ei = min(N, round((anoms(k,2)+anoms(k,3))*fs));
    raw_bytes = raw_bytes + (ei-si)*4;
end
fprintf('Raw bytes for anomaly windows (per-channel sums): %d bytes\n', raw_bytes);

% reconstruct from kp (cubic interp) and from quantized packed (simulate)
recon_kp = zeros(6,N); recon_q = zeros(6,N);
mse_kp = zeros(6,1); mse_q = zeros(6,1);
for ch=1:6
    idx = kp_idx_g{ch};
    if numel(idx) < 4
        recon_kp(ch,:) = air2(ch,:);
        recon_q(ch,:) = air2(ch,:);
    else
        recon_kp(ch,:) = interp1(idx, air2(ch,idx), 1:N, 'pchip', 'extrap');
        % quantized values: replace flagged idx with quantized values then interp
        vals = air2(ch, idx);
        f = flag_kp{ch};
        for i=1:numel(idx)
            if f(i)
                % find packed entry for this (ch-1, idx(i))
                % find matching event
                % (slow loop OK for demo)
                for j=1:evcnt
                    if events(j).ch == ch-1 && events(j).kpos == idx(i)
                        q = round(events(j).amp_d*scale_amp);
                        q = max(-32,min(31,q));
                        vals(i) = q/scale_amp;
                        break;
                    end
                end
            end
        end
        recon_q(ch,:) = interp1(idx, vals, 1:N, 'pchip', 'extrap');
    end
    mse_kp(ch) = mean((recon_kp(ch,:) - air2(ch,:)).^2);
    mse_q(ch)  = mean((recon_q(ch,:) - air2(ch,:)).^2);
end

disp('MSE per channel (kp full / quantized):');
for ch=1:6
    fprintf('%s: %.6e / %.6e, kp_count=%d\n', names{ch}, mse_kp(ch), mse_q(ch), numel(kp_idx_g{ch}));
end

% simple plotting (similar to python)
figure('Position',[100 100 1000 900]);
for ch=1:6
    subplot(6,1,ch);
    plot(t, gx(ch,:), 'k'); hold on;
    plot(t, air2(ch,:), 'b');
    plot(t, recon_kp(ch,:), 'r');
    plot(t, recon_q(ch,:), 'g');
    idx = kp_idx_g{ch};
    scatter(idx/fs, air2(ch, idx), 20, 'r', 'filled');
    f = flag_kp{ch};
    if sum(f)>0
        scatter((idx(f))/fs, air2(ch, idx(f)), 80, 'm');
    end
    ylabel(names{ch});
    if ch==1, legend('ground','air','recon_kp','recon_q','kp','flagged'); end
end
xlabel('time (s)'); sgtitle('Pack demo (MATLAB)');
