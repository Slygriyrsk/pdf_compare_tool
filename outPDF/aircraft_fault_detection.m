clear; clc; close all;

%% Aircraft model
A = [-0.313 56.7 0;
    -0.0139 -0.426 0;
     0       56.7  0];
B = [0.232; 0.0203; 0];
C = [0 0 1];

A_gnd = A; B_gnd = B; C_gnd = C;
Ts = 0.01;

sysd   = c2d(ss(A,B,C,0), Ts);
Ad = sysd.A; Bd = sysd.B; Cd = sysd.C;

sysd_g = c2d(ss(A_gnd,B_gnd,C_gnd,0), Ts);
Ad_g = sysd_g.A; Bd_g = sysd_g.B; Cd_g = sysd_g.C;

N       = 1200;
t       = (0:N-1)*Ts;
rad2deg = 180/pi;

fault_start = 500;
fault_end   = 700;
sigma_sensor = 0.005;

rng(1);
Q = 1e-5 * eye(3);
R = 1e-4;

Xair    = zeros(3,1);
Xground = zeros(3,1);
Pg      = eye(3);
u       = 0.1 * ones(1, N);

%% Signal limits
n_bits    = 7;
alpha_lim = [-0.5  0.5];
q_lim     = [-1    1  ];
theta_lim = [-0.5  0.5];
de_lim    = [-0.3  0.3];

%% Storage
x_air    = zeros(3, N);  x_ground = zeros(3, N);
alpha_q  = zeros(1, N);  q_q      = zeros(1, N);
theta_q  = zeros(1, N);  u_q      = zeros(1, N);
gnd_innovation = zeros(1,N);  residual   = zeros(1,N);
quant_err_y    = zeros(1,N);  y_dequantize = zeros(1,N);
store_k        = zeros(3,N);  flag       = zeros(1,N);
Splus_hist     = zeros(1,N);  Sminus_hist  = zeros(1,N);
packed_word    = zeros(1, N, 'uint32');

pred_alpha = 0; pred_q = 0; pred_th = 0; pred_de = 0;

k_cusum = 0.02; h_cusum = 0.15;
Splus = 0; Sminus = 0; flag_count = 0; M = 5;

innov_buffer = zeros(1,10);
R_original   = R;

%% ---- MAIN LOOP ----
for k = 1:N
    if k >= fault_start && k <= fault_end
        theta_F    = Xair(3) + 0.05;
        u_physical = u(k);
    else
        theta_F    = Xair(3);
        u_physical = u(k);
    end

    Xair = Ad * Xair + Bd * u_physical;
    Yair = Cd * Xair;

    %% DPCM encode -> 7-bit signed codes [-64..63]
    code_a  = dpcm_enc(Xair(1),    pred_alpha, alpha_lim, n_bits);
    code_q  = dpcm_enc(Xair(2),    pred_q,     q_lim,     n_bits);
    code_th = dpcm_enc(theta_F,    pred_th,    theta_lim, n_bits);
    code_de = dpcm_enc(u_physical, pred_de,    de_lim,    n_bits);

    %% PACK into uint32
    %  Bit layout (0-based LSB=0):
    %    [6:0]   = de      [13:7]  = theta
    %    [20:14] = q       [27:21] = alpha    [31:28] = unused
    %  Add offset=64 to make signed codes unsigned before packing
    offset = 2^(n_bits-1);
    ua = uint32(code_a  + offset);
    uq = uint32(code_q  + offset);
    ut = uint32(code_th + offset);
    ud = uint32(code_de + offset);

    word = bitor(bitor(bitshift(ua,21), bitshift(uq,14)), ...
                 bitor(bitshift(ut, 7), ud));
    packed_word(k) = word;

    %% UNPACK from uint32
    mask7 = uint32(127);
    ua2 = bitand(bitshift(word,-21), mask7);
    uq2 = bitand(bitshift(word,-14), mask7);
    ut2 = bitand(bitshift(word, -7), mask7);
    ud2 = bitand(word,               mask7);

    code_a2  = double(ua2) - offset;
    code_q2  = double(uq2) - offset;
    code_th2 = double(ut2) - offset;
    code_de2 = double(ud2) - offset;

    %% DPCM decode
    [aq,  pred_alpha] = dpcm_dec(code_a2,  pred_alpha, alpha_lim, n_bits);
    [qq,  pred_q]     = dpcm_dec(code_q2,  pred_q,     q_lim,     n_bits);
    [thq, pred_th]    = dpcm_dec(code_th2, pred_th,    theta_lim, n_bits);
    [deq, pred_de]    = dpcm_dec(code_de2, pred_de,    de_lim,    n_bits);

    alpha_q(k) = aq;  q_q(k) = qq;  theta_q(k) = thq;  u_q(k) = deq;
    yq = thq;  y_dequantize(k) = yq;

    %% Kalman filter (ground)
    X_pred = Ad_g * Xground + Bd_g * u(k);
    Ppred  = Ad_g * Pg * Ad_g' + Q;
    innov  = yq - Cd_g * X_pred;

    innov_buffer     = [innov_buffer(2:end), innov];
    moving_innov_var = var(innov_buffer);
    theoretical_var  = Cd_g * Ppred * Cd_g' + R_original;

    if moving_innov_var > theoretical_var
        R_adaptive = moving_innov_var;
    else
        R_adaptive = R_original;
    end

    K       = Ppred * Cd_g' / (Cd_g * Ppred * Cd_g' + R_adaptive);
    Xground = X_pred + K * innov;
    Pg      = (eye(3) - K * Cd_g) * Ppred;

    store_k(:,k)      = K;
    x_air(:,k)        = Xair;
    x_ground(:,k)     = Xground;
    gnd_innovation(k) = innov;
    residual(k)       = abs(innov);
    quant_err_y(k)    = Yair - yq;

    %% CUSUM
    Splus  = max(0, Splus  + ( innov - k_cusum));
    Sminus = max(0, Sminus + (-innov - k_cusum));
    Splus_hist(k) = Splus;  Sminus_hist(k) = Sminus;

    if Splus > h_cusum || Sminus > h_cusum
        flag_count = flag_count + 1;
    else
        flag_count = 0;
    end
    if flag_count >= M,  flag(k) = 1;  end
end

%% ---- PLOTS ----

figure(1);
plot(t,x_air(1,:),'k','LineWidth',1.2); hold on;
plot(t,alpha_q,'r--','LineWidth',1);
plot(t,x_air(2,:),'b','LineWidth',1.2);
plot(t,q_q,'m--','LineWidth',1);
plot(t,x_air(3,:),'g','LineWidth',1.2);
plot(t,theta_q,'c--','LineWidth',1);
xlabel('Time (s)'); ylabel('Value');
legend('\alpha_{true}','\alpha_q','q_{true}','q_q','\theta_{true}','\theta_q');
title('Plot 1: Quantized vs True Signals (DPCM)'); grid on;

figure(2);
plot(t,u,'k','LineWidth',1.2); hold on;
plot(t,u_q,'r--','LineWidth',1);
xlabel('Time (s)'); ylabel('\delta_e');
legend('True Input','Quantized Input');
title('Plot 2: Elevator Input - True vs Quantized'); grid on;

figure(3);
air_innov_signal = x_air(3,:) - y_dequantize;
plot(t,air_innov_signal,'-k','LineWidth',1); hold on;
plot(t,gnd_innovation,'--r','LineWidth',1);
xlabel('Time (s)'); ylabel('Innovation (rad)');
legend('Air (Quantization Error)','Ground (KF Residual)');
title('Plot 3: Innovation Signals'); grid on;

figure(4);
detect_threshold = 3*sqrt(R);
plot(t,residual,'b','LineWidth',1.2); hold on;
yline(detect_threshold,'r--','Threshold','LineWidth',1.5);
xlabel('Time (s)'); ylabel('|Innovation|');
title('Plot 4: Fault Detection (KF Residual)'); grid on;

figure(5);
plot(t,x_air(3,:)*rad2deg,'-k','LineWidth',1.2); hold on;
plot(t,x_ground(3,:)*rad2deg,'--r','LineWidth',1.2);
xlabel('Time (s)'); ylabel('\theta (deg)');
legend('Air (True)','Ground (KF Estimate)');
title('Plot 5: State Estimation (\theta)'); grid on;

figure(6);
plot(t,(x_air(2,:)-x_ground(2,:))*rad2deg,'LineWidth',1.2);
xlabel('Time (s)'); ylabel('q_{air} - q_{ground} (deg/s)');
title('Plot 6: Direct Fault (Air - Ground Pitch Rate)'); grid on;

figure(7);
plot(t,Splus_hist,'b','LineWidth',1.2); hold on;
plot(t,Sminus_hist,'r','LineWidth',1.2);
yline(h_cusum,'k--','CUSUM Threshold','LineWidth',1.5);
xlabel('Time (s)'); ylabel('CUSUM Statistic');
legend('S^+','S^-','Threshold');
title('Plot 7: CUSUM Statistics'); grid on;

figure(8);
plot(t,store_k(1,:),'r','LineWidth',1.2); hold on;
plot(t,store_k(2,:),'g','LineWidth',1.2);
plot(t,store_k(3,:),'b','LineWidth',1.2);
xlabel('Time (s)'); ylabel('Gain Magnitude');
legend('K_{\alpha}','K_q','K_{\theta}');
title('Plot 8: Kalman Gains Over Time'); grid on;

figure(9);
subplot(2,1,1);
plot(t,x_air(3,:)*rad2deg,'-k','LineWidth',1.2); hold on;
plot(t,x_ground(3,:)*rad2deg,'--r','LineWidth',1.2);
ylabel('\theta (deg)'); legend('True','KF Estimate');
title('State Estimation Performance'); grid on;
subplot(2,1,2);
plot(t,(x_air(2,:)-x_ground(2,:))*rad2deg,'m','LineWidth',1.2);
ylabel('\Delta q (deg/s)'); xlabel('Time (s)');
legend('Estimation Error (q)'); grid on;

%% ---- LOCAL FUNCTIONS ----

function code = dpcm_enc(x, pred, lim, nb)
    step = (lim(2)-lim(1))/(2^nb-1);
    code = round((x-pred)/step);
    code = max(-2^(nb-1), min(2^(nb-1)-1, code));
end

function [xq, pred_out] = dpcm_dec(code, pred, lim, nb)
    step     = (lim(2)-lim(1))/(2^nb-1);
    xq       = pred + code*step;
    xq       = max(lim(1), min(lim(2), xq));
    pred_out = xq;
end
