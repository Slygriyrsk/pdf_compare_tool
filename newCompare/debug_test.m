%% ============================================================
%  debug_test.m  - Run each block independently in Command Window
%  Run section by section using Ctrl+Enter (Run Section)
%% ============================================================

%% BLOCK 1: Test DPCM encode/decode roundtrip
% Expected: decoded value close to original (small quant error)
nb  = 7;
lim = [-0.5 0.5];
x_test  = 0.123;
pred    = 0.0;

step = (lim(2)-lim(1))/(2^nb-1);
code = round((x_test-pred)/step);
code = max(-2^(nb-1), min(2^(nb-1)-1, code));
xq   = pred + code*step;
xq   = max(lim(1), min(lim(2), xq));

fprintf('--- BLOCK 1: DPCM roundtrip ---\n');
fprintf('Original  : %.6f\n', x_test);
fprintf('Code      : %d\n',   code);
fprintf('Decoded   : %.6f\n', xq);
fprintf('Error     : %.6f\n', abs(x_test - xq));
% PASS if Error < step/2 = %.6f
fprintf('Max allowed error (step/2): %.6f\n\n', step/2);

%% BLOCK 2: Test pack32 / unpack32
% Pack 4 signals into one uint32 and unpack them back
% Expected: all codes_out == codes_in
n_bits = 7;
offset = 2^(n_bits-1);   % 64

ca = 5;   cq = -10;   ct = 20;   cd = -3;   % example signed codes
ua = uint32(ca+offset);
uq = uint32(cq+offset);
ut = uint32(ct+offset);
ud = uint32(cd+offset);

word = bitor(bitor(bitshift(ua,21), bitshift(uq,14)), ...
             bitor(bitshift(ut, 7), ud));

mask7 = uint32(127);
ua2 = double(bitand(bitshift(word,-21),mask7)) - offset;
uq2 = double(bitand(bitshift(word,-14),mask7)) - offset;
ut2 = double(bitand(bitshift(word, -7),mask7)) - offset;
ud2 = double(bitand(word,              mask7))  - offset;

fprintf('--- BLOCK 2: Pack/Unpack ---\n');
fprintf('alpha : in=%3d  out=%3d  %s\n', ca, ua2, check_eq(ca,ua2));
fprintf('q     : in=%3d  out=%3d  %s\n', cq, uq2, check_eq(cq,uq2));
fprintf('theta : in=%3d  out=%3d  %s\n', ct, ut2, check_eq(ct,ut2));
fprintf('de    : in=%3d  out=%3d  %s\n', cd, ud2, check_eq(cd,ud2));
fprintf('Packed word (binary): %s\n\n', dec2bin(word,32));

%% BLOCK 3: Check CUSUM logic
% Inject artificial positive innovations and check S+ grows
fprintf('--- BLOCK 3: CUSUM logic ---\n');
k_c = 0.02;  h_c = 0.15;
Sp = 0; Sm = 0;
innov_seq = [0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05];
for i = 1:length(innov_seq)
    Sp = max(0, Sp + (innov_seq(i) - k_c));
    Sm = max(0, Sm + (-innov_seq(i) - k_c));
end
fprintf('After 8 steps of innov=0.05:\n');
fprintf('S+ = %.4f  (should exceed threshold %.2f: %s)\n', Sp, h_c, yn(Sp>h_c));
fprintf('S- = %.4f\n\n', Sm);

%% BLOCK 4: Check discrete model stability
% Eigenvalues of Ad should be inside unit circle
A = [-0.313 56.7 0; -0.0139 -0.426 0; 0 56.7 0];
B = [0.232; 0.0203; 0]; C = [0 0 1];
Ts = 0.01;
sysd = c2d(ss(A,B,C,0), Ts);
Ad   = sysd.A;
ev   = eig(Ad);
fprintf('--- BLOCK 4: Discrete model eigenvalues ---\n');
for i = 1:length(ev)
    fprintf('eig %d: |lambda|=%.6f  (stable if <1: %s)\n', i, abs(ev(i)), yn(abs(ev(i))<1));
end
fprintf('\n');

%% BLOCK 5: Bandwidth calculation
fprintf('--- BLOCK 5: Bandwidth ---\n');
fs  = 1/Ts;
bps_dpcm = n_bits * 4 * fs;   % 4 signals x 7 bits x 100 Hz
bps_raw  = 32 * 4 * fs;       % 4 signals x 32 bits x 100 Hz
fprintf('DPCM BW      : %d bits/sec\n', bps_dpcm);
fprintf('Raw float BW : %d bits/sec\n', bps_raw);
fprintf('BW reduction : %.1f %%\n', 100*(1 - bps_dpcm/bps_raw));

%% ---- helper functions (must be at end of script) ----
function s = check_eq(a,b)
    if a==b, s='PASS'; else, s='FAIL'; end
end
function s = yn(cond)
    if cond, s='YES'; else, s='NO'; end
end
