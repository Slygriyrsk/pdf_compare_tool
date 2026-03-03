% Quick check: value preserved after DPCM quantize -> dequantize?

n_bits = 7;
lims   = {[-0.5 0.5], [-1 1], [-0.5 0.5], [-0.3 0.3]};
names  = {'alpha', 'q', 'theta', 'de'};
test_vals = [0.123, -0.456, 0.234, -0.187];   % example values

fprintf('%-8s  %-10s  %-10s  %-12s  %-6s\n','Signal','Original','Decoded','Error','Status');
fprintf('%s\n', repmat('-',1,52));

for i = 1:4
    lim  = lims{i};
    x    = test_vals(i);
    step = (lim(2)-lim(1))/(2^n_bits-1);
    
    % encode
    code = round(x / step);
    code = max(-2^(n_bits-1), min(2^(n_bits-1)-1, code));
    
    % decode
    xq = code * step;
    xq = max(lim(1), min(lim(2), xq));
    
    err = abs(x - xq);
    ok  = err <= step/2;
    fprintf('%-8s  %-10.5f  %-10.5f  %-12.2e  %s\n', ...
        names{i}, x, xq, err, status(ok));
end

fprintf('\nMax quantization step sizes:\n');
for i = 1:4
    lim  = lims{i};
    step = (lim(2)-lim(1))/(2^n_bits-1);
    fprintf('  %-8s : step = %.5f  (%.4f%%  of range)\n', ...
        names{i}, step, 100*step/(lim(2)-lim(1)));
end

fprintf('\nCompression: 32-bit float -> 7-bit DPCM\n');
fprintf('Ratio : 7/32 = %.3f  (%.1f%% bandwidth saving)\n', 7/32, 100*(1-7/32));

function s = status(ok)
    if ok, s = 'OK'; else, s = 'WARN'; end
end
