clear; close all; clc;

%% IMPROVED CONFIGURATION
RAW_SIGNAL_SIZE = 512;
INPUT_SIZE = RAW_SIGNAL_SIZE/2;
HIDDEN_SIZE = 1024;  % Increased from 512
OUTPUT_SIZE = 7;

TRAIN_SAMPLES = 15000;  % Increased from 5000
EPOCHS = 3000;  % Increased from 2000
LR = 0.008;  % Reduced for better convergence
BATCH = 64;  % Reduced from 128 for more frequent updates
FS = 1024;

rng(42);  % Different seed for better initialization

act_list = {'relu', 'tanh', 'pre_relu'};
results = struct();

%% GENERATE IMPROVED DATASET
fprintf('Generating training dataset...\n');
X = zeros(TRAIN_SAMPLES, INPUT_SIZE);
Y = zeros(TRAIN_SAMPLES, OUTPUT_SIZE);

for i = 1:TRAIN_SAMPLES
    [x, y] = gen_sample_improved(FS, RAW_SIGNAL_SIZE, OUTPUT_SIZE);
    X(i, :) = x;
    Y(i, :) = y;
    if mod(i, 1000) == 0
        fprintf('Generated %d/%d samples\n', i, TRAIN_SAMPLES);
    end
end

%% IMPROVED NORMALIZATION
% Use log scale for FFT magnitudes (better for frequency domain)
X_log = log10(X + 1e-8);
mu = mean(X_log, 1);
sigma = std(X_log, 0, 1) + 1e-6;
X_norm = (X_log - mu) ./ sigma;

%% TRAIN NETWORKS
for aidx = 1:numel(act_list)
    act = act_list{aidx};
    fprintf('\n=== Training with %s activation ===\n', act);
    
    rng(42);
    [W1, b1, W2, b2, W3, b3] = init_weights_improved(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    if strcmp(act, 'pre_relu')
        alpha = ones(HIDDEN_SIZE, 1) * 0.01;
    else
        alpha = [];
    end
    
    [W1, b1, W2, b2, W3, b3, alpha, history] = train_nn_improved(...
        X_norm, Y, W1, b1, W2, b2, W3, b3, alpha, FS, ...
        'epochs', EPOCHS, 'lr', LR, 'batch', BATCH, 'act', act);
    
    results.(act).W1 = W1;
    results.(act).b1 = b1;
    results.(act).W2 = W2;
    results.(act).b2 = b2;
    results.(act).W3 = W3;
    results.(act).b3 = b3;
    results.(act).alpha = alpha;
    results.(act).history = history;
    results.(act).mu = mu;
    results.(act).sigma = sigma;
end

%% PLOT TRAINING CURVES
figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);
colors = lines(numel(act_list));

% Loss plot
subplot(2, 2, 1);
hold on;
for i = 1:numel(act_list)
    act = act_list{i};
    plot(results.(act).history.loss, 'Color', colors(i, :), 'LineWidth', 1.5);
end
legend(act_list, 'Location', 'best');
xlabel('Epoch');
ylabel('Loss (Hz²)');
title('Training Loss');
grid on;
set(gca, 'YScale', 'log');

% Accuracy plot (1 Hz threshold)
subplot(2, 2, 2);
hold on;
for i = 1:numel(act_list)
    act = act_list{i};
    plot(results.(act).history.acc_1hz, 'Color', colors(i, :), 'LineWidth', 1.5);
end
legend(act_list, 'Location', 'best');
xlabel('Epoch');
ylabel('Accuracy');
title('Accuracy (predictions within 1 Hz)');
grid on;

% Mean absolute error plot
subplot(2, 2, 3);
hold on;
for i = 1:numel(act_list)
    act = act_list{i};
    plot(results.(act).history.mae, 'Color', colors(i, :), 'LineWidth', 1.5);
end
legend(act_list, 'Location', 'best');
xlabel('Epoch');
ylabel('MAE (Hz)');
title('Mean Absolute Error');
grid on;

% Accuracy plot (5 Hz threshold for comparison)
subplot(2, 2, 4);
hold on;
for i = 1:numel(act_list)
    act = act_list{i};
    plot(results.(act).history.acc_5hz, 'Color', colors(i, :), 'LineWidth', 1.5);
end
legend(act_list, 'Location', 'best');
xlabel('Epoch');
ylabel('Accuracy');
title('Accuracy (predictions within 5 Hz)');
grid on;

%% TEST ON NEW SAMPLES
ntest = 10;  % More test samples
fprintf('\n=== Testing on new samples ===\n');

figure('Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
for i = 1:numel(act_list)
    act = act_list{i};
    all_true = [];
    all_pred = [];
    errors = [];
    
    for k = 1:ntest
        [x_raw, y] = gen_sample_improved(FS, RAW_SIGNAL_SIZE, OUTPUT_SIZE);
        
        % Apply same preprocessing
        x_log = log10(x_raw + 1e-8);
        x_norm = (x_log - results.(act).mu) ./ results.(act).sigma;
        
        [~, ~, ~, ~, out] = forward_pass(x_norm, results.(act).W1, ...
            results.(act).b1, results.(act).W2, results.(act).b2, ...
            results.(act).W3, results.(act).b3, act, results.(act).alpha);
        
        true_hz = y * (FS/2);
        pred_hz = out' * (FS/2);
        
        all_true = [all_true true_hz];
        all_pred = [all_pred pred_hz];
        errors = [errors abs(pred_hz - true_hz)];
    end
    
    % Calculate metrics
    mae = mean(errors);
    rmse = sqrt(mean(errors.^2));
    acc_1hz = sum(errors <= 1) / numel(errors);
    acc_5hz = sum(errors <= 5) / numel(errors);
    
    % Plot
    subplot(2, ceil(numel(act_list)/2), i);
    scatter(all_true, all_pred, 30, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    mn = min([all_true all_pred]);
    mx = max([all_true all_pred]);
    plot([mn mx], [mn mx], 'r--', 'LineWidth', 2);
    xlabel('True Frequency (Hz)');
    ylabel('Predicted Frequency (Hz)');
    title(sprintf('%s\nMAE: %.2f Hz, Acc@1Hz: %.1f%%', ...
        act, mae, acc_1hz*100));
    grid on;
    axis equal;
    xlim([mn mx]);
    ylim([mn mx]);
    
    fprintf('%s: MAE=%.3f Hz, RMSE=%.3f Hz, Acc@1Hz=%.2f%%, Acc@5Hz=%.2f%%\n', ...
        act, mae, rmse, acc_1hz*100, acc_5hz*100);
end

%% HELPER FUNCTIONS

function [x_fft, y] = gen_sample_improved(fs, signal_size, output_size)
    % IMPROVED: Wider frequency range and better signal generation
    minf = 50;   % Changed from 400
    maxf = 500;  % Changed from 407 (wider range)
    
    % Ensure frequencies are well-separated (at least 5 Hz apart)
    freqs = zeros(1, output_size);
    freqs(1) = minf + rand * (maxf - minf - 5*output_size);
    for k = 2:output_size
        freqs(k) = freqs(k-1) + 5 + rand * 10;  % 5-15 Hz spacing
    end
    
    amps = 0.3 + rand(1, output_size) * 1.2;  % Wider amplitude range
    phases = rand(1, output_size) * 2 * pi;
    
    % Normalize output to [0, 1]
    y = freqs / (fs/2);
    
    % Generate time-domain signal
    n = 0:(signal_size-1);
    t = n / fs;
    x_raw = zeros(1, signal_size);
    
    for k = 1:output_size
        x_raw = x_raw + amps(k) * sin(2*pi*freqs(k)*t + phases(k));
    end
    
    % Add realistic noise (lower SNR)
    noise_level = 0.05 * std(x_raw);
    x_raw = x_raw + randn(1, signal_size) * noise_level;
    
    % Compute FFT
    sig_fft = fft(x_raw);
    mag_fft = abs(sig_fft(1:signal_size/2));
    
    % Improved normalization: preserve relative magnitudes
    x_fft = mag_fft / (max(mag_fft) + 1e-8);
end

function [W1, b1, W2, b2, W3, b3] = init_weights_improved(D, H, C)
    % Xavier/Glorot initialization
    scale1 = sqrt(2.0 / (D + H));
    scale2 = sqrt(2.0 / (H + H));
    scale3 = sqrt(2.0 / (H + C));
    
    W1 = (rand(H, D) * 2 - 1) * scale1;
    b1 = zeros(H, 1);
    
    W2 = (rand(H, H) * 2 - 1) * scale2;
    b2 = zeros(H, 1);
    
    W3 = (rand(C, H) * 2 - 1) * scale3;
    b3 = zeros(C, 1);
end

function [z1, a1, z2, a2, out] = forward_pass(x, W1, b1, W2, b2, W3, b3, act, alpha)
    x = x(:);
    
    % Layer 1
    z1 = W1 * x + b1;
    a1 = apply_act(z1, act, alpha);
    
    % Layer 2
    z2 = W2 * a1 + b2;
    a2 = apply_act(z2, act, alpha);
    
    % Output layer (no activation, linear output)
    out = W3 * a2 + b3;
    
    % Clip output to valid range [0, 1]
    out = max(0, min(1, out));
end

function a = apply_act(z, act, alpha)
    switch act
        case 'tanh'
            a = tanh(z);
        case 'relu'
            a = max(0, z);
        case 'pre_relu'
            a = max(0, z) + alpha .* min(0, z);
    end
end

function da = apply_act_derivative(z, act, alpha)
    switch act
        case 'tanh'
            da = 1 - tanh(z).^2;
        case 'relu'
            da = double(z > 0);
        case 'pre_relu'
            mask = double(z > 0);
            da = mask + (1 - mask) .* alpha;
    end
end

function L = hz_mse_loss(pred, target, fs)
    % Loss in Hz space (more interpretable)
    pred_hz = pred * (fs/2);
    target_hz = target * (fs/2);
    d = pred_hz - target_hz;
    L = mean(d.^2);
end

function [W1, b1, W2, b2, W3, b3, alpha, history] = train_nn_improved(...
    X, Y, W1, b1, W2, b2, W3, b3, alpha, fs, varargin)
    
    p = inputParser;
    addParameter(p, 'epochs', 3000);
    addParameter(p, 'lr', 0.008);
    addParameter(p, 'batch', 64);
    addParameter(p, 'act', 'relu');
    parse(p, varargin{:});
    
    EPOCHS = p.Results.epochs;
    LR = p.Results.lr;
    BATCH = p.Results.batch;
    act_type = p.Results.act;
    
    N = size(X, 1);
    C = size(Y, 2);
    H = size(W1, 1);
    
    % Initialize history
    history.loss = zeros(EPOCHS, 1);
    history.mae = zeros(EPOCHS, 1);
    history.acc_1hz = zeros(EPOCHS, 1);
    history.acc_5hz = zeros(EPOCHS, 1);
    
    % Learning rate schedule
    lr_current = LR;
    
    for epoch = 1:EPOCHS
        % Adjust learning rate
        if mod(epoch, 1000) == 0
            lr_current = lr_current * 0.5;
            fprintf('  Adjusted LR to %.6f\n', lr_current);
        end
        
        % Shuffle data
        idx = randperm(N);
        epoch_loss = 0;
        
        % Mini-batch training
        for bstart = 1:BATCH:N
            bidx = idx(bstart:min(bstart+BATCH-1, N));
            bsize = numel(bidx);
            
            % Initialize gradients
            dW3 = zeros(size(W3));
            db3 = zeros(size(b3));
            dW2 = zeros(size(W2));
            db2 = zeros(size(b2));
            dW1 = zeros(size(W1));
            db1 = zeros(size(b1));
            
            if strcmp(act_type, 'pre_relu')
                dalpha = zeros(H, 1);
            end
            
            % Accumulate gradients over batch
            for bi = 1:bsize
                i = bidx(bi);
                x = X(i, :);
                target = Y(i, :)';
                
                % Forward pass
                [z1, a1, z2, a2, out] = forward_pass(x, W1, b1, W2, b2, ...
                    W3, b3, act_type, alpha);
                
                % Compute loss in Hz space
                epoch_loss = epoch_loss + hz_mse_loss(out, target, fs);
                
                % Backward pass (in normalized space)
                dout = (2/C) * (out - target);
                
                % Output layer
                db3 = db3 + dout;
                dW3 = dW3 + dout * a2';
                
                % Hidden layer 2
                da2 = W3' * dout;
                dz2 = da2 .* apply_act_derivative(z2, act_type, alpha);
                db2 = db2 + dz2;
                dW2 = dW2 + dz2 * a1';
                
                % Hidden layer 1
                da1 = W2' * dz2;
                dz1 = da1 .* apply_act_derivative(z1, act_type, alpha);
                db1 = db1 + dz1;
                dW1 = dW1 + dz1 * x(:)';
                
                % PReLU alpha gradient
                if strcmp(act_type, 'pre_relu')
                    mask1 = double(z1 <= 0);
                    dalpha = dalpha + (mask1 .* (dz1 .* z1));
                    
                    mask2 = double(z2 <= 0);
                    dalpha = dalpha + (mask2 .* (dz2 .* z2));
                end
            end
            
            % Update weights
            inv = 1.0 / bsize;
            W3 = W3 - lr_current * dW3 * inv;
            b3 = b3 - lr_current * db3 * inv;
            W2 = W2 - lr_current * dW2 * inv;
            b2 = b2 - lr_current * db2 * inv;
            W1 = W1 - lr_current * dW1 * inv;
            b1 = b1 - lr_current * db1 * inv;
            
            if strcmp(act_type, 'pre_relu')
                alpha = alpha - lr_current * dalpha * inv;
            end
        end
        
        % Record epoch metrics
        history.loss(epoch) = epoch_loss / N;
        
        % Compute accuracy on random subset
        sel = randperm(N, min(500, N));
        Xv = X(sel, :);
        Yv = Y(sel, :);
        [acc_1hz, acc_5hz, mae] = compute_accuracy_improved(...
            W1, b1, W2, b2, W3, b3, alpha, Xv, Yv, fs, act_type);
        
        history.acc_1hz(epoch) = acc_1hz;
        history.acc_5hz(epoch) = acc_5hz;
        history.mae(epoch) = mae;
        
        % Print progress
        if mod(epoch, 100) == 0
            fprintf('Epoch %d: loss=%.6f, MAE=%.3f Hz, Acc@1Hz=%.2f%%, Acc@5Hz=%.2f%%\n', ...
                epoch, history.loss(epoch), mae, acc_1hz*100, acc_5hz*100);
        end
    end
end

function [acc_1hz, acc_5hz, mae] = compute_accuracy_improved(...
    W1, b1, W2, b2, W3, b3, alpha, X, Y, fs, act)
    
    N = size(X, 1);
    C = size(Y, 2);
    cnt_1hz = 0;
    cnt_5hz = 0;
    total_error = 0;
    
    for i = 1:N
        [~, ~, ~, ~, out] = forward_pass(X(i, :), W1, b1, W2, b2, ...
            W3, b3, act, alpha);
        
        pred_hz = out * (fs/2);
        true_hz = Y(i, :)' * (fs/2);
        
        diffs = abs(pred_hz - true_hz);
        cnt_1hz = cnt_1hz + sum(diffs <= 1);
        cnt_5hz = cnt_5hz + sum(diffs <= 5);
        total_error = total_error + sum(diffs);
    end
    
    acc_1hz = cnt_1hz / (N * C);
    acc_5hz = cnt_5hz / (N * C);
    mae = total_error / (N * C);
end
