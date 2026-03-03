% DPCM encode -> decode error check
% Tests whether value is preserved through the full DPCM roundtrip

n_bits = 7;
lims  = {[-0.5 0.5], [-1 1], [-0.5 0.5], [-0.3 0.3]};
names = {'alpha', 'q', 'theta', 'de'};

% Simulate a short sequence per signal (pred carries over sample to sample)
test_seq = [0.0, 0.05, 0.10, 0.12, 0.11, 0.09];  % slowly varying like real aircraft

fprintf('DPCM Encode -> Decode Error Check\n');
fprintf('%s\n', repmat('-',1,60));

for i = 1:4
    lim  = lims{i};
    step = (lim(2)-lim(1)) / (2^n_bits - 1);
    pred = 0;  max_err = 0;

    fprintf('Signal: %s\n', names{i});
    fprintf('  %-6s  %-10s  %-10s  %-12s\n','Sample','Original','Decoded','Error');

    for k = 1:length(test_seq)
        x = test_seq(k);

        % DPCM encode
        code = round((x - pred) / step);
        code = max(-2^(n_bits-1), min(2^(n_bits-1)-1, code));

        % DPCM decode
        xq   = pred + code * step;
        xq   = max(lim(1), min(lim(2), xq));
        pred = xq;  % update predictor with decoded value

        err     = abs(x - xq);
        max_err = max(max_err, err);
        fprintf('  k=%-4d  %-10.5f  %-10.5f  %-12.2e\n', k, x, xq, err);
    end
    fprintf('  Max error: %.2e  (step=%.2e)  --> %s\n\n', ...
        max_err, step, tf(max_err <= step/2));
end

function s = tf(c)
    if c, s='PRESERVED (within half-step)'; else, s='WARN: exceeds half-step'; end
end
