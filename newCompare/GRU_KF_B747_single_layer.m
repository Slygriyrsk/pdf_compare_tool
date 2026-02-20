%% B747 State Estimation: Kalman Filter + Single-Layer GRU Correction
% Same logic as the two-layer version but with ONE GRU layer only.
%
% Architecture:
%   Input  : [x_kf (5x1); udot (2x1)]  => nx=7
%   GRU    : nh hidden units (single layer)
%   Output : tanh(Wy*h + by)            => ny=5  (normalised residual)
%   x_gru  : Ad*x_kf(:,k) + Bd*u(:,k) + dx   (anchored to KF, not self-propagating)
%
% All fixes from the two-layer version are kept:
%   - C measurement matrix defined explicitly
%   - GRU output clamped with tanh (prevents integrator drift)
%   - x_gru anchored to x_kf (not Ad*x_gru) so errors stay bounded
%   - BPTT: dh computed before weight updates, uses h_prev for recurrent grads
%   - Gradient clipping (global norm threshold = 1.0)
%   - Input normalisation computed once, outside epoch loop
%   - Per-state y-axis scaling in plots so all three lines are visible

clear; clc; close all;

%% =========================================================
%  SIMULATION PARAMETERS
%% =========================================================
Ts = 0.01;          % sampling time (s)
T  = 400;           % total time (s)
N  = round(T/Ts);   % number of samples = 40000

%% =========================================================
%  B747 MODEL PARAMETERS
%% =========================================================
Xu  = -1.982e3;
Xw  =  4.025e3;
Zu  = -2.595e4;
Zw  = -9.030e4;
Zq  = -4.524e5;
Zwd =  1.909e3;
Mu  =  1.593e4;
Mw  = -1.563e5;
Mq  = -1.521e7;
Mwd = -1.702e4;
g   =  9.81;
S    = 511;          % gross wing area (m^2)
cbar = 8.324;        % mean aerodynamic chord (m)
U0   = 235.9;        % nominal forward speed (m/s)
Iyy  = 0.449e8;      % pitch moment of inertia (kg.m^2)
m    = 2.83176e6/g;  % aircraft mass (kg)
rho  = 0.3045;       % air density (kg/m^3)

% Control derivatives
Xdp = 0.3*m*g;
Zdp = 0;
Mdp = 0;
Xde = -3.818e-6 * (0.5*rho*U0^2*S);
Zde = -0.3648   * (0.5*rho*U0^2*S);
Mde = -1.444    * (0.5*rho*U0^2*S*cbar);

%% =========================================================
%  CONTINUOUS STATE-SPACE  (5 states: u, w, q, theta, h)
%% =========================================================
A = [ Xu/m,                          Xw/m,                              0,                                   -g;
      Zu/(m-Zwd),                    Zw/(m-Zwd),                        (Zq+m*U0)/(m-Zwd),                   0;
      (Mu+Zu*Mwd/(m-Zwd))/Iyy,      (Mw+Zw*Mwd/(m-Zwd))/Iyy,          (Mq+(Zq+m*U0)*Mwd/(m-Zwd))/Iyy,    0;
      0,                             0,                                 1,                                   0 ];

% Altitude augmentation: h_dot = -w + U0*theta  (small-angle approximation)
A_aug = [ A,          zeros(4,1);
          0, -1, 0, U0, 0 ];

B_aug = [ Xde/m,                          Xdp/m;
          Zde/(m-Zwd),                    Zdp/(m-Zwd);
          (Mde+Zde*Mwd/(m-Zwd))/Iyy,     (Mdp+Zdp*Mwd/(m-Zwd))/Iyy;
          0,                             0;
          0,                             0 ];

Bw_aug = [ -Xu/m,                        -Xw/m,                          0;
           -Zu/(m-Zwd),                  -Zw/(m-Zwd),                    0;
           (-Mu-Zu*Mwd/(m-Zwd))/Iyy,    (-Mw-Zw*Mwd/(m-Zwd))/Iyy,     -Mq/Iyy;
            0,                            0,                              0;
            0,                            0,                              0 ];

%% =========================================================
%  DISCRETISE (zero-order hold via c2d)
%% =========================================================
sysd = c2d(ss(A_aug, [B_aug, Bw_aug], eye(5), 0), Ts);
Ad   = sysd.A;
Bd   = sysd.B(:, 1:2);
Bwd  = sysd.B(:, 3:5);

%% =========================================================
%  MEASUREMENT MATRIX  (u, w, q, theta measured; h is not)
%% =========================================================
C = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0 ];

%% =========================================================
%  STORAGE
%% =========================================================
x_true = zeros(5, N);
x_kf   = zeros(5, N);
x_gru  = zeros(5, N);

x_true(:,1) = [250; 50; 0.2; 0.1; 2000];
x_kf(:,1)   = x_true(:,1);
x_gru(:,1)  = x_true(:,1);

%% =========================================================
%  CONTROL INPUTS  (random walk in elevator + throttle)
%% =========================================================
u    = zeros(2, N);
udot = zeros(2, N);
for k = 2:N
    udot(:,k) = 0.01 * randn(2,1);
    u(:,k)    = u(:,k-1) + Ts*udot(:,k);
end

%% =========================================================
%  KALMAN FILTER NOISE COVARIANCES
%% =========================================================
Q = diag([1e-10, 1e-10, 1e-11, 1e-13, 1e-13]);   % process noise
R = diag([(1e-1)^2, (1e-1)^2, (1e-1)^2, (1e-2)^2]); % measurement noise
P = eye(5);

%% =========================================================
%  KALMAN FILTER SIMULATION  (generates x_true and x_kf)
%% =========================================================
for k = 2:N
    % Sinusoidal wind gust active between 100s and 200s
    wg = zeros(3,1);
    if (k*Ts) >= 100 && (k*Ts) <= 200
        wg(1) = 10  * sin(2*pi*(k*Ts));
        wg(2) =  5  * cos(2*pi*(k*Ts));
        wg(3) =  1.5;
    end

    % True state propagation (with gust)
    x_true(:,k) = Ad*x_true(:,k-1) + Bd*u(:,k-1) + Bwd*wg;

    % Noisy measurement of [u; w; q; theta]
    y = C*x_true(:,k) + 0.02*randn(4,1);

    % KF predict
    x_pred = Ad*x_kf(:,k-1) + Bd*u(:,k-1);
    P_pred = Ad*P*Ad' + Q;

    % KF update
    K_gain    = P_pred*C' / (C*P_pred*C' + R);
    x_kf(:,k) = x_pred + K_gain*(y - C*x_pred);
    P         = (eye(5) - K_gain*C) * P_pred;
end

%% =========================================================
%  SINGLE-LAYER GRU SETUP
%% =========================================================
sig      = @(x) 1./(1+exp(-x));   % sigmoid
sig_der  = @(f) f.*(1-f);         % sigmoid derivative (in terms of output)
tanh_der = @(f) 1 - f.^2;         % tanh derivative (in terms of output)

nx = 7;    % input size  : 5 KF states + 2 control rates
nh = 16;   % hidden size : single GRU layer  (use 16 to match total capacity of old 2-layer)
ny = 5;    % output size : 5 state corrections

lr = 1e-4; % learning rate

% ---- Xavier initialisation ----
lim = sqrt(6 / (nh + nx));
Wr = -lim + 2*lim*rand(nh, nx);   % reset gate   - input weight
Ur = -lim + 2*lim*rand(nh, nh);   % reset gate   - recurrent weight
br = zeros(nh, 1);                 % reset gate   - bias

Wz = -lim + 2*lim*rand(nh, nx);   % update gate  - input weight
Uz = -lim + 2*lim*rand(nh, nh);   % update gate  - recurrent weight
bz = zeros(nh, 1);                 % update gate  - bias

Wh = -lim + 2*lim*rand(nh, nx);   % candidate    - input weight
Uh = -lim + 2*lim*rand(nh, nh);   % candidate    - recurrent weight
bh = zeros(nh, 1);                 % candidate    - bias

Wy = 0.01*randn(ny, nh);           % output layer - weight
by = zeros(ny, 1);                 % output layer - bias

%% =========================================================
%  INPUT NORMALISATION  ([-1,1] min-max, computed once)
%% =========================================================
all_inputs = zeros(nx, N-2);
for kk = 2:N-1
    all_inputs(:, kk-1) = [x_kf(:,kk); udot(:,kk)];
end
xmin = min(all_inputs, [], 2);
xmax = max(all_inputs, [], 2);
xrng = xmax - xmin;
xrng(xrng < 1e-8) = 1e-8;   % avoid divide-by-zero on near-constant channels

% Scale factor that converts normalised GRU output back to physical units
% Max expected one-step residual for each state
target_scale = [50; 50; 0.1; 0.1; 1000];

%% =========================================================
%  GRU TRAINING  (online, single-step truncated BPTT)
%% =========================================================
epochs = 100;

for ep = 1:epochs

    h          = zeros(nh, 1);   % hidden state reset at start of each epoch
    total_loss = 0;
    total_err  = zeros(ny, 1);

    for k = 2:N-1

        % ---------- Normalise input ----------
        inp      = [x_kf(:,k); udot(:,k)];
        inp_norm = 2*(inp - xmin)./xrng - 1;   % in [-1, 1]

        % ---------- Save previous hidden state (needed for BPTT) ----------
        h_prev = h;

        % ==========================================================
        %  FORWARD PASS
        % ==========================================================

        % Reset gate
        r = sig(Wr*inp_norm + Ur*h_prev + br);          % (nh x 1)

        % Update gate
        z = sig(Wz*inp_norm + Uz*h_prev + bz);          % (nh x 1)

        % Candidate hidden state  (reset gate applied to h_prev)
        h_tilde = tanh(Wh*inp_norm + Uh*(r.*h_prev) + bh);  % (nh x 1)

        % New hidden state (convex combination gated by z)
        h_new = (1-z).*h_prev + z.*h_tilde;             % (nh x 1)

        % Output: linear layer followed by tanh to keep it in (-1, 1)
        % The tanh is ESSENTIAL - without it the correction dx = y_hat*scale
        % is unbounded and the altitude integrator (eigenvalue=1) causes
        % x_gru to diverge to millions of metres within 40000 steps.
        y_hat_lin = Wy*h_new + by;                       % (ny x 1)
        y_hat     = tanh(y_hat_lin);                     % (ny x 1)  in (-1,1)

        % ---------- Training target ----------
        % GRU is trained to predict the one-step residual that the KF misses:
        %   y_true = [x_true(k+1) - KF_prediction(k+1)] / target_scale
        % Normalising by target_scale keeps the target in a similar range to y_hat.
        % We also clip y_true to [-1,1] for consistency with the output range.
        y_true = (x_true(:,k+1) - (Ad*x_kf(:,k) + Bd*u(:,k))) ./ target_scale;
        y_true = max(-1, min(1, y_true));

        % Loss = sum of squared errors
        e          = y_hat - y_true;                     % (ny x 1)
        total_err  = total_err + abs(e);
        total_loss = total_loss + e'*e;

        % ==========================================================
        %  BACKWARD PASS  (truncated BPTT, single time step)
        %  Reference: Chung et al. (2014), GRU equations;
        %             University of Pennsylvania BPTT-GRU tutorial
        % ==========================================================

        % -- Gradient through output tanh --
        % dL/d(y_hat_lin) = e .* (1 - y_hat^2)
        delta_out = e .* (1 - y_hat.^2);                % (ny x 1)

        % -- Output layer weight gradients --
        dWy = delta_out * h_new';                        % (ny x nh)
        dby = delta_out;                                 % (ny x 1)

        % -- Gradient flowing back into h_new --
        % dL/d(h_new) = Wy' * delta_out
        dh = Wy' * delta_out;                            % (nh x 1)

        % -- Update gate gradient --
        % h_new = (1-z).*h_prev + z.*h_tilde
        % d(h_new)/dz = h_tilde - h_prev   (element-wise)
        dz = dh .* (h_tilde - h_prev) .* sig_der(z);   % (nh x 1)

        % -- Candidate hidden state gradient --
        % d(h_new)/d(h_tilde) = z   (element-wise)
        dh_tilde = dh .* z .* tanh_der(h_tilde);        % (nh x 1)

        % -- Reset gate gradient --
        % h_tilde = tanh(Wh*x + Uh*(r.*h_prev) + bh)
        % d(h_tilde)/dr flows through Uh*(r.*h_prev) term:
        %   d(h_tilde_pre)/dr = Uh' * dh_tilde  (backprop through Uh)
        %   then element-wise multiply by h_prev  (chain rule for r.*h_prev)
        %   then multiply by sigmoid derivative of r
        dr = (Uh' * dh_tilde) .* h_prev .* sig_der(r);  % (nh x 1)

        % -- Gradient clipping (global norm, threshold = 1.0) --
        % Prevents exploding gradients, especially in early epochs before
        % the GRU has learned to produce small, stable corrections.
        all_grads   = [dWy(:); dby; dz; dh_tilde; dr];
        gnorm       = norm(all_grads);
        clip_thresh = 1.0;
        if gnorm > clip_thresh
            scale_g  = clip_thresh / gnorm;
            dWy      = dWy      * scale_g;
            dby      = dby      * scale_g;
            dz       = dz       * scale_g;
            dh_tilde = dh_tilde * scale_g;
            dr       = dr       * scale_g;
        end

        % -- Weight updates (SGD) --
        % Output layer
        Wy = Wy - lr * dWy;
        by = by - lr * dby;

        % Update gate weights
        % NOTE: Uz uses h_prev (the state that was INPUT to this time step),
        % not h_new (the state that was OUTPUT). Using h_new here is a common
        % BPTT mistake - it computes gradients w.r.t. the wrong time index.
        Wz = Wz - lr * (dz * inp_norm');
        Uz = Uz - lr * (dz * h_prev');    % recurrent: outer product with h_prev
        bz = bz - lr * dz;

        % Candidate weights
        % The recurrent term in h_tilde is Uh*(r.*h_prev), so the gradient
        % w.r.t. Uh involves (r.*h_prev), and uses h_prev not h_new.
        Wh = Wh - lr * (dh_tilde * inp_norm');
        Uh = Uh - lr * (dh_tilde * (r.*h_prev)');
        bh = bh - lr * dh_tilde;

        % Reset gate weights
        Wr = Wr - lr * (dr * inp_norm');
        Ur = Ur - lr * (dr * h_prev');    % recurrent: outer product with h_prev
        br = br - lr * dr;

        % -- Advance hidden state --
        h = h_new;
    end

    fprintf('Epoch %3d | Loss = %.6f | Error norm = %.6f\n', ...
        ep, total_loss/(N-2), norm(total_err/(N-2)));
end

%% =========================================================
%  GRU INFERENCE PASS  (build x_gru using trained weights)
%% =========================================================
h = zeros(nh, 1);   % reset hidden state

for k = 2:N-1

    % Normalise input (same scheme as training)
    inp      = [x_kf(:,k); udot(:,k)];
    inp_norm = 2*(inp - xmin)./xrng - 1;

    % Forward pass (identical structure to training)
    r       = sig(  Wr*inp_norm + Ur*h  + br );
    z       = sig(  Wz*inp_norm + Uz*h  + bz );
    h_tilde = tanh( Wh*inp_norm + Uh*(r.*h)  + bh );
    h       = (1-z).*h + z.*h_tilde;

    y_hat = tanh(Wy*h + by);         % bounded in (-1, 1)
    dx    = y_hat .* target_scale;   % scale back to physical units

    % CORRECT INFERENCE FORMULA - anchored to x_kf, not self-propagating:
    %
    %   x_gru(:,k+1) = Ad*x_kf(:,k) + Bd*u(:,k) + dx
    %
    % This matches exactly what the GRU was trained to predict:
    %   y_true = x_true(k+1) - [Ad*x_kf(k) + Bd*u(k)]
    %
    % If we instead used Ad*x_gru(:,k), the GRU would propagate its OWN
    % state. Any residual GRU error would then be fed back through the
    % dynamics. Since the altitude state has eigenvalue = 1.0 (pure
    % integrator), even a tiny constant bias of 5 m/step accumulates to
    % 5 x 40000 = 200,000 m over the full simulation, completely dominating
    % the plot and making x_true and x_kf appear flat at "zero".
    x_gru(:,k+1) = Ad*x_kf(:,k) + Bd*u(:,k) + dx;
end
x_gru(:,N) = x_gru(:,N-1);   % fill the last sample

%% =========================================================
%  RMSE ACCURACY
%% =========================================================
fprintf('\n-------- RMSE Accuracy --------\n');
state_labels = {'u', 'w', 'q', 'theta', 'h'};
idx = 2:N;
for j = 1:5
    rmse_kf  = sqrt(mean((x_true(j,idx) - x_kf(j,idx)).^2));
    rmse_gru = sqrt(mean((x_true(j,idx) - x_gru(j,idx)).^2));
    fprintf('%5s  |  KF RMSE = %8.3f  |  GRU RMSE = %8.3f\n', ...
        state_labels{j}, rmse_kf, rmse_gru);
end

%% =========================================================
%  PLOTTING
%% =========================================================
t      = (0:N-1)*Ts;
labels = {'u (m/s)', 'w (m/s)', 'q (rad/s)', '\theta (rad)', 'h (m)'};

figure('Name', 'B747 - KF + Single-Layer GRU', ...
       'NumberTitle', 'off', 'Position', [100 50 950 750]);

for i = 1:5
    subplot(5, 1, i);
    plot(t, x_true(i,:), 'k',   'LineWidth', 1.2); hold on;
    plot(t, x_kf(i,:),   'r--', 'LineWidth', 1.0);
    plot(t, x_gru(i,:),  'b-.', 'LineWidth', 1.0);
    ylabel(labels{i});
    grid on;

    % Lock y-axis to x_true/x_kf range so x_gru cannot dominate the scale.
    % If the GRU correction is larger than the KF range you will see it
    % extend beyond the axis, which is informative rather than misleading.
    ydata = [x_true(i,:), x_kf(i,:)];
    ylo   = min(ydata);
    yhi   = max(ydata);
    ypad  = max((yhi - ylo)*0.15, 1e-3);
    ylim([ylo - ypad, yhi + ypad]);

    if i == 1
        legend('True', 'KF', 'KF + GRU', 'Location', 'best');
    end
end
xlabel('Time (s)');
sgtitle('B747 Longitudinal State Estimation — Single-Layer GRU');
