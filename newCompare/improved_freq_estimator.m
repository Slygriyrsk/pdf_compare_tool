clear; close all; clc;

% =========================================================================
% NETWORK ARCHITECTURE PARAMETERS
% =========================================================================
% RAW_SIGNAL_SIZE: Total time-domain samples (must be power of 2 for FFT)
% When to change: Keep as power of 2 (256, 512, 1024). Larger = more frequency resolution
RAW_SIGNAL_SIZE = 512;

% INPUT_SIZE: FFT magnitude features fed to network (half of raw signal due to FFT symmetry)
% When to change: Automatically set to RAW_SIGNAL_SIZE/2
INPUT_SIZE = RAW_SIGNAL_SIZE/2;

% HIDDEN_SIZE: Number of neurons in hidden layers
% When to change: Increase (256->512->1024) if model underfits, decrease if overfits
% Current: 512 neurons
HIDDEN_SIZE = 512;

% OUTPUT_SIZE: Number of sine waves to predict (7 frequencies)
% When to change: Match the number of frequencies you want to detect
OUTPUT_SIZE = 7;

% =========================================================================
% TRAINING HYPERPARAMETERS
% =========================================================================
% TRAIN_SAMPLES: Number of training examples
% When to change: Increase (5000->10000) if underfitting, decrease if training is too slow
TRAIN_SAMPLES = 5000;

% EPOCHS: Number of complete passes through training data
% When to change: Increase if loss still decreasing, decrease if overfitting starts
EPOCHS = 2000;

% LR (Learning Rate): Step size for weight updates
% When to change: Decrease (0.016->0.008) if training is unstable/loss oscillates
%                 Increase (0.016->0.032) if learning is too slow
%                 Try: 0.001, 0.005, 0.01, 0.016, 0.02, 0.05
LR = 0.016;

% BATCH: Mini-batch size for gradient descent
% When to change: Larger batch (128->256) = more stable but slower
%                 Smaller batch (128->64->32) = faster but noisier gradients
BATCH = 128;

% FS: Sampling frequency in Hz
% When to change: Must be > 2*max_frequency (Nyquist). Don't change unless signal changes
FS = 1024;

rng(0); % Fixed random seed for reproducibility

act_list = {'relu','tanh','pre_relu'};
results = struct();

fprintf('========================================\n');
fprintf('NEURAL NETWORK CONFIGURATION\n');
fprintf('========================================\n');
fprintf('Input Size (FFT features): %d\n', INPUT_SIZE);
fprintf('Hidden Layer Size: %d neurons\n', HIDDEN_SIZE);
fprintf('Output Size (frequencies): %d\n', OUTPUT_SIZE);
fprintf('Training Samples: %d\n', TRAIN_SAMPLES);
fprintf('Epochs: %d\n', EPOCHS);
fprintf('Learning Rate: %.4f\n', LR);
fprintf('Batch Size: %d\n', BATCH);
fprintf('Sampling Frequency: %d Hz\n', FS);
fprintf('========================================\n\n');

% =========================================================================
% GENERATE DATASET
% =========================================================================
fprintf('Generating training dataset...\n');
X = zeros(TRAIN_SAMPLES, INPUT_SIZE);
Y = zeros(TRAIN_SAMPLES, OUTPUT_SIZE);

% Generate first sample to show what's happening
fprintf('\n--- EXAMPLE OF FIRST TRAINING SAMPLE ---\n');
[x_example, y_example] = gen_sample(FS, RAW_SIGNAL_SIZE, OUTPUT_SIZE);
fprintf('Input: FFT magnitude vector (length %d)\n', length(x_example));
fprintf('First 10 FFT values: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', x_example(1:10));
fprintf('\nOutput: Normalized frequencies (length %d)\n', length(y_example));
fprintf('Normalized freqs: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', y_example);
fprintf('Actual Hz: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n', y_example*(FS/2));
fprintf('----------------------------------------\n\n');

for i = 1:TRAIN_SAMPLES
    [x, y] = gen_sample(FS, RAW_SIGNAL_SIZE, OUTPUT_SIZE);
    X(i, :) = x;
    Y(i, :) = y;
    if mod(i, 1000) == 0
        fprintf('Generated %d/%d samples\n', i, TRAIN_SAMPLES);
    end
end

fprintf('Dataset generation complete!\n\n');

% =========================================================================
% NORMALIZATION
% =========================================================================
% Normalize input features (zero mean, unit variance)
% When to change: Always normalize! Helps convergence
mu = mean(X, 1);
sigma = std(X, 0, 1) + 1e-6;  % Add small epsilon to avoid division by zero
X = (X - mu) ./ sigma;

fprintf('Input normalization complete (mean=0, std=1)\n\n');

% =========================================================================
% TRAIN MODELS WITH DIFFERENT ACTIVATIONS
% =========================================================================
for aidx = 1:numel(act_list)
    act = act_list{aidx};
    fprintf('========================================\n');
    fprintf('TRAINING WITH ACTIVATION: %s\n', upper(act));
    fprintf('========================================\n');
    
    rng(0);  % Reset seed for fair comparison
    [W1, b1, W2, b2, W3, b3] = init_weights(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    
    % PReLU needs learnable alpha parameters
    if strcmp(act, 'pre_relu')
        alpha = ones(HIDDEN_SIZE, 1) * 0.01;  % Small negative slope initially
    else
        alpha = [];
    end
    
    [W1, b1, W2, b2, W3, b3, alpha, history] = train_nn(X, Y, W1, b1, W2, b2, W3, b3, alpha, FS, ...
        'epochs', EPOCHS, 'lr', LR, 'batch', BATCH, 'act', act);
    
    results.(act).W1 = W1; results.(act).b1 = b1;
    results.(act).W2 = W2; results.(act).b2 = b2;
    results.(act).W3 = W3; results.(act).b3 = b3;
    results.(act).alpha = alpha;
    results.(act).history = history;
    
    fprintf('\n%s Training Complete!\n', upper(act));
    fprintf('Final Loss: %.6f\n', history.loss(end));
    fprintf('Final Accuracy: %.3f\n\n', history.acc(end));
end

% =========================================================================
% PLOT TRAINING CURVES
% =========================================================================
figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.6]);
subplot(2, 1, 1); hold on;
colors = lines(numel(act_list));
for i = 1:numel(act_list)
    act = act_list{i};
    plot(results.(act).history.loss, 'Color', colors(i, :), 'LineWidth', 1.5);
end
legend(act_list);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss (MSE)');
grid on;

subplot(2, 1, 2); hold on;
for i = 1:numel(act_list)
    act = act_list{i};
    plot(results.(act).history.acc, 'Color', colors(i, :), 'LineWidth', 1.5);
end
legend(act_list);
xlabel('Epoch');
ylabel('Accuracy');
title('Accuracy (fraction of predictions within 1 Hz)');
grid on;

% =========================================================================
% TEST ON NEW SAMPLES
% =========================================================================
ntest = 5;
fprintf('========================================\n');
fprintf('TESTING ON %d NEW SAMPLES\n', ntest);
fprintf('========================================\n\n');

figure('Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
for i = 1:numel(act_list)
    act = act_list{i};
    all_true = []; all_pred = [];
    
    fprintf('--- Testing with %s activation ---\n', upper(act));
    for k = 1:ntest
        [x, y] = gen_sample(FS, INPUT_SIZE, OUTPUT_SIZE);
        
        % Normalize test input using training statistics
        x = (x - mu) ./ sigma;
        
        [z1, a1, z2, a2, out] = forward_pass(x, results.(act).W1, results.(act).b1, ...
            results.(act).W2, results.(act).b2, results.(act).W3, results.(act).b3, act, results.(act).alpha);
        
        % Convert to Hz
        true_hz = y * (FS/2);
        pred_hz = out' * (FS/2);
        all_true = [all_true true_hz];
        all_pred = [all_pred pred_hz];
        
        % Print sample results
        fprintf('Test %d:\n', k);
        fprintf('  True Hz: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n', true_hz);
        fprintf('  Pred Hz: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n', pred_hz);
        fprintf('  Errors:  [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f] Hz\n\n', abs(true_hz - pred_hz));
    end
    
    % Calculate mean absolute error
    mae = mean(abs(all_true - all_pred));
    fprintf('%s Mean Absolute Error: %.3f Hz\n\n', upper(act), mae);
    
    % Plot scatter
    subplot(2, ceil(numel(act_list)/2), i);
    scatter(all_true, all_pred, 30, 'filled'); hold on;
    mn = min([all_true all_pred]);
    mx = max([all_true all_pred]);
    plot([mn mx], [mn mx], 'r--', 'LineWidth', 2);
    xlabel('True Hz');
    ylabel('Predicted Hz');
    title(sprintf('%s (MAE=%.2f Hz)', act, mae));
    grid on;
    axis equal;
    xlim([mn mx]);
    ylim([mn mx]);
end

fprintf('========================================\n');
fprintf('PARAMETER TUNING GUIDE\n');
fprintf('========================================\n');
fprintf('IF LOSS IS HIGH (not decreasing):\n');
fprintf('  - Increase HIDDEN_SIZE (512->1024)\n');
fprintf('  - Increase EPOCHS (2000->5000)\n');
fprintf('  - Decrease LR if oscillating (0.016->0.008)\n');
fprintf('  - Try different activation (tanh often works well)\n\n');
fprintf('IF LOSS DECREASES THEN PLATEAUS:\n');
fprintf('  - Increase TRAIN_SAMPLES (5000->10000)\n');
fprintf('  - Add more EPOCHS\n');
fprintf('  - Try learning rate decay\n\n');
fprintf('IF PREDICTIONS ARE WILDLY OFF:\n');
fprintf('  - Check frequency range (400-407 Hz) fits in FS/2\n');
fprintf('  - Increase INPUT_SIZE for better frequency resolution\n');
fprintf('  - Normalize outputs (Y) if not already\n\n');
fprintf('BEST PRACTICES:\n');
fprintf('  - Start with LR=0.01, adjust by factors of 2\n');
fprintf('  - HIDDEN_SIZE should be 1-4x INPUT_SIZE\n');
fprintf('  - BATCH: 32-256 (larger=stable, smaller=fast)\n');
fprintf('  - Always use normalization!\n');
fprintf('========================================\n');

% =========================================================================
% FUNCTIONS
% =========================================================================

function [x_fft, y] = gen_sample(fs, input_size, output_size)
    % Generate mixed sine wave with random frequencies between 400-407 Hz
    minf = 400;
    maxf = 407;
    freqs = sort(minf + rand(1, output_size) * (maxf - minf));
    amps = 0.5 + rand(1, output_size) * 0.8;  % Random amplitudes [0.5, 1.3]
    phases = rand(1, output_size) * 2 * pi;    % Random phases [0, 2π]
    
    % Normalize frequencies to [0, 1] range
    y = freqs / (fs/2);
    
    % Generate time-domain signal
    x_raw = zeros(1, input_size);
    for n = 0:input_size-1
        t = n / fs;
        v = 0;
        for k = 1:output_size
            v = v + amps(k) * sin(2*pi*freqs(k)*t + phases(k));
        end
        v = v + (rand - 0.5) * 0.02;  % Add small noise
        x_raw(n+1) = v;
    end
    
    % Compute FFT and take magnitude of positive frequencies
    sig_fft = fft(x_raw);
    mag_fft = abs(sig_fft(1:input_size/2));
    x_fft = mag_fft / max(mag_fft + 1e-6);  % Normalize to [0, 1]
end

function [W1, b1, W2, b2, W3, b3] = init_weights(D, H, C)
    % He initialization for better convergence with ReLU
    scale1 = sqrt(2.0 / D);
    scale2 = sqrt(2.0 / H);
    W1 = (rand(H, D) * 2 - 1) * scale1;
    b1 = zeros(H, 1);
    W2 = (rand(H, H) * 2 - 1) * scale2;
    b2 = zeros(H, 1);
    W3 = (rand(C, H) * 2 - 1) * scale2;
    b3 = zeros(C, 1);
end

function [z1, a1, z2, a2, out] = forward_pass(x, W1, b1, W2, b2, W3, b3, act, alpha)
    x = x(:);  % Ensure column vector
    
    % Layer 1
    z1 = W1 * x + b1;
    a1 = apply_act(z1, act, alpha);
    
    % Layer 2
    z2 = W2 * a1 + b2;
    a2 = apply_act(z2, act, alpha);
    
    % Output layer (no activation)
    out = W3 * a2 + b3;
end

function a = apply_act(z, act, alpha)
    switch act
        case 'tanh'
            a = tanh(z);
        case 'relu'
            a = max(0, z);
        case 'pre_relu'
            a = max(0, z) + alpha .* min(0, z);  % Parametric ReLU
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
            da = mask + (1 - mask) .* alpha;  % 1 if z>0, alpha if z<=0
    end
end

function L = mse_loss(pred, target)
    d = pred - target(:);
    L = mean(d.^2);  % Mean Squared Error
end

function [W1, b1, W2, b2, W3, b3, alpha, history] = train_nn(X, Y, W1, b1, W2, b2, W3, b3, alpha, fs, varargin)
    p = inputParser;
    addParameter(p, 'epochs', 2000);
    addParameter(p, 'lr', 0.016);
    addParameter(p, 'batch', 16);
    addParameter(p, 'act', 'relu');
    parse(p, varargin{:});
    
    EPOCHS = p.Results.epochs;
    LR = p.Results.lr;
    BATCH = p.Results.batch;
    act_type = p.Results.act;
    
    N = size(X, 1);
    C = size(Y, 2);
    H = size(W1, 1);
    
    history.loss = zeros(EPOCHS, 1);
    history.acc = zeros(EPOCHS, 1);
    
    for epoch = 1:EPOCHS
        idx = randperm(N);
        epoch_loss = 0;
        
        for bstart = 1:BATCH:N
            bidx = idx(bstart:min(bstart+BATCH-1, N));
            bsize = numel(bidx);
            
            % Initialize gradient accumulators
            dW3 = zeros(size(W3)); db3 = zeros(size(b3));
            dW2 = zeros(size(W2)); db2 = zeros(size(b2));
            dW1 = zeros(size(W1)); db1 = zeros(size(b1));
            if strcmp(act_type, 'pre_relu')
                dalpha = zeros(H, 1);
            end
            
            % Accumulate gradients over mini-batch
            for bi = 1:bsize
                i = bidx(bi);
                x = X(i, :);
                target = Y(i, :)';
                
                % Forward pass
                [z1, a1, z2, a2, out] = forward_pass(x, W1, b1, W2, b2, W3, b3, act_type, alpha);
                
                % Compute loss
                epoch_loss = epoch_loss + mse_loss(out, target);
                
                % Backward pass (compute gradients)
                dout = (2/C) * (out - target);  % Gradient of MSE
                
                % Output layer gradients
                db3 = db3 + dout;
                dW3 = dW3 + dout * a2';
                
                % Hidden layer 2 gradients
                da2 = W3' * dout;
                dz2 = da2 .* apply_act_derivative(z2, act_type, alpha);
                db2 = db2 + dz2;
                dW2 = dW2 + dz2 * a1';
                
                % Hidden layer 1 gradients
                da1 = W2' * dz2;
                dz1 = da1 .* apply_act_derivative(z1, act_type, alpha);
                db1 = db1 + dz1;
                dW1 = dW1 + dz1 * x(:)';
                
                % PReLU alpha gradients
                if strcmp(act_type, 'pre_relu')
                    mask1 = double(z1 <= 0);
                    dalpha = dalpha + (mask1 .* z1 .* dz1);
                    
                    mask2 = double(z2 <= 0);
                    dalpha = dalpha + (mask2 .* z2 .* dz2);
                end
            end
            
            % Update weights with averaged gradients
            inv = 1.0 / bsize;
            W3 = W3 - LR * dW3 * inv;
            b3 = b3 - LR * db3 * inv;
            W2 = W2 - LR * dW2 * inv;
            b2 = b2 - LR * db2 * inv;
            W1 = W1 - LR * dW1 * inv;
            b1 = b1 - LR * db1 * inv;
            
            if strcmp(act_type, 'pre_relu')
                alpha = alpha - LR * dalpha * inv;
            end
        end
        
        % Record metrics
        history.loss(epoch) = epoch_loss / N;
        
        % Compute accuracy on random subset
        sel = randperm(N, min(200, N));
        Xv = X(sel, :);
        Yv = Y(sel, :);
        history.acc(epoch) = compute_accuracy(W1, b1, W2, b2, W3, b3, alpha, Xv, Yv, fs, act_type);
        
        if mod(epoch, 100) == 0
            fprintf('Epoch %d/%d, Loss=%.6f, Acc=%.3f\n', epoch, EPOCHS, history.loss(epoch), history.acc(epoch));
        end
    end
end

function acc = compute_accuracy(W1, b1, W2, b2, W3, b3, alpha, X, Y, fs, act)
    N = size(X, 1);
    C = size(Y, 2);
    cnt = 0;
    
    for i = 1:N
        [~, ~, ~, ~, out] = forward_pass(X(i, :), W1, b1, W2, b2, W3, b3, act, alpha);
        
        % Convert to Hz and compute errors
        pred_hz = out * (fs/2);
        true_hz = Y(i, :)' * (fs/2);
        diffs = abs(pred_hz - true_hz);
        cnt = cnt + sum(diffs <= 1);  % Count predictions within 1 Hz
    end
    
    acc = cnt / (N * C);
end
