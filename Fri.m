clear; clc; close all;

%% System
A = [-0.313 56.7 0;
     -0.0139 -0.426 0;
      0     56.7  0];

B = [0.232; 0.0203; 0];
C = [0 0 1];

Ts = 0.01;
sysd = c2d(ss(A,B,C,0),Ts);
Ad = sysd.A; Bd = sysd.B; Cd = sysd.C;

%% Simulation
N = 1200;
t = (0:N-1)*Ts;

fault_start = 500;
fault_end   = 700;
fault_mag   = 0.15;

Q = 1e-5*eye(3);
R = 1e-4;

x = zeros(3,1);
xhat_g = zeros(3,1);
Pg = eye(3);

u = 0.05*sin(0.5*t);

%% Limits
alpha_lim = [-0.5 0.5];
q_lim     = [-1 1];
theta_lim = [-0.5 0.5];
de_lim    = [-0.3 0.3];

%% Storage
x_air = zeros(3,N);
x_ground = zeros(3,N);
innovation = zeros(1,N);
residual = zeros(1,N);

alpha_q = zeros(1,N);
q_q = zeros(1,N);
theta_q = zeros(1,N);
u_q = zeros(1,N);

%% Loop
for k = 1:N

    x = Ad*x + Bd*u(k);

    if k >= fault_start && k <= fault_end
        x(2) = x(2) + fault_mag;
    end

    y = Cd*x;

    aq  = quant7(x(1),alpha_lim);
    qq  = quant7(x(2),q_lim);
    thq = quant7(x(3),theta_lim);
    deq = quant7(u(k),de_lim);

    alpha_q(k) = aq;
    q_q(k) = qq;
    theta_q(k) = thq;
    u_q(k) = deq;

    word = pack32(aq,qq,thq,deq,0);
    [~,qq,~,deq,~] = unpack32(word);

    yq = dequant7(qq,q_lim);
    uq = dequant7(deq,de_lim);

    xpred = Ad*xhat_g + Bd*uq;
    Ppred = Ad*Pg*Ad' + Q;

    innov = yq - Cd*xpred;
    K = Ppred*Cd'/(Cd*Ppred*Cd' + R);

    xhat_g = xpred + K*innov;
    Pg = (eye(3) - K*Cd)*Ppred;

    x_air(:,k) = x;
    x_ground(:,k) = xhat_g;
    innovation(k) = innov;
    residual(k) = abs(innov);
end

disp('================ QUANTIZED VALUES ================')
disp('Quantized y (pitch rate) sample:')
disp(q_q(480:520))

disp('Quantized u (elevator) sample:')
disp(u_q(480:520))

disp('================ QUANTIZATION ERROR ================')
fprintf('Mean |y - y_q| = %.6f\n',mean(abs(quant_err_y)))
fprintf('Mean |u - u_q| = %.6f\n',mean(abs(quant_err_u)))

disp('================ BANDWIDTH =================')
fprintf('Telemetry BW = %d bits/sec\n',BW_util)
fprintf('Raw BW (4 signals @ 32-bit) = %d bits/sec\n',4*32*fs)
fprintf('BW Reduction = %.2f %%\n',100*(1 - BW_util/(4*32*fs)))


%% ================== PLOTS ==================

%% Plot 1: Quantized Signals
figure(1);
plot(t,alpha_q,t,q_q,t,theta_q,'LineWidth',1);
xlabel('Time (s)');
ylabel('Quantized Value');
legend('\alpha_q','q_q','\theta_q');
title('Plot 1: Quantized Signals');
grid on;

%% Plot 2: Input
figure(2);
plot(t,u,'k','LineWidth',1.2); hold on;
plot(t,u_q,'r--','LineWidth',1);
xlabel('Time (s)');
ylabel('\delta_e');
legend('True Input','Quantized Input');
title('Plot 2: Elevator Input');
grid on;

%% Plot 5: Innovation
figure(5);
plot(t,innovation,'LineWidth',1.2);
xlabel('Time (s)');
ylabel('Innovation');
title('Plot 5: Innovation Signal');
grid on;

%% Plot 6: Fault Detection
figure(6);
threshold = 3*std(innovation);
plot(t,residual,'LineWidth',1.2); hold on;
yline(threshold,'r--','Threshold');
xlabel('Time (s)');
ylabel('|Innovation|');
title('Plot 6: Fault Detection');
grid on;

%% Plot 7: State Estimation
figure(7);
plot(t,x_air(3,:),'-k','LineWidth',1.2); hold on;
plot(t,x_ground(3,:),'--r','LineWidth',1.2);
xlabel('Time (s)');
ylabel('\theta');
legend('Air (True)','Ground (Estimated)');
title('Plot 7: State Estimation');
grid on;

%% Plot 9: Direct Fault (Air - Ground)
figure(9);
plot(t,x_air(2,:) - x_ground(2,:),'LineWidth',1.2);
xlabel('Time (s)');
ylabel('q_{air} - q_{ground}');
title('Plot 9: Direct Fault (Air - Ground)');
grid on;

%% ================== FUNCTIONS ==================

function q = quant7(x,lim)
    L = 2^7 - 1;
    x = min(max(x,lim(1)),lim(2));
    q = round((x-lim(1))/(lim(2)-lim(1))*L);
end

function x = dequant7(q,lim)
    L = 2^7 - 1;
    x = q/L*(lim(2)-lim(1)) + lim(1);
end

function word = pack32(a,q,th,de,flags)
    word = uint32(0);
    word = bitor(word, bitshift(uint32(a),25));
    word = bitor(word, bitshift(uint32(q),18));
    word = bitor(word, bitshift(uint32(th),11));
    word = bitor(word, bitshift(uint32(de),4));
    word = bitor(word, uint32(flags));
end

function [a,q,th,de,flags] = unpack32(word)
    a     = bitand(bitshift(word,-25),127);
    q     = bitand(bitshift(word,-18),127);
    th    = bitand(bitshift(word,-11),127);
    de    = bitand(bitshift(word,-4),127);
    flags = bitand(word,15);
end
