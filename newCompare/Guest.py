clear; clc; close all;

%% =============================
%  B747 LONGITUDINAL MODEL
% ==============================

g = 9.81; theta0 = 0; U0 = 235.9;
m = 2.83176e6 / g; Iyy = 0.449e8;
rho = 0.3045; S = 511; cbar = 8.324;

Xu=-1.982e3; Xw=4.025e3; Zu=-2.595e4; Zw=-9.030e4;
Zq=-4.524e5; Zwd=1.909e3; Mu=1.593e4; Mw=-1.563e5;
Mq=-1.521e7; Mwd=-1.702e4;

A = [ Xu/m, Xw/m, 0, -g;
      Zu/(m-Zwd), Zw/(m-Zwd), (Zq+m*U0)/(m-Zwd), 0;
      (Mu+Zu*Mwd/(m-Zwd))/Iyy, ...
      (Mw+Zw*Mwd/(m-Zwd))/Iyy, ...
      (Mq+(Zq+m*U0)*Mwd/(m-Zwd))/Iyy, 0;
      0 0 1 0];

A_aug = [A zeros(4,1);
         0 -1 0 U0 0];

B_aug = zeros(5,2);
Bw_aug = zeros(5,3);

Ts = 0.01;
sysd = c2d(ss(A_aug,[B_aug Bw_aug],eye(5),0),Ts);
Ad = sysd.A;
Bd = sysd.B(:,1:2);
Bwd = sysd.B(:,3:5);

%% =============================
%  TRAINING DATA GENERATION
% ==============================

numSim = 40;
N = 1500;
t = (0:N-1)*Ts;

XTrain = cell(numSim,1);
YTrain = cell(numSim,1);

for s=1:numSim
    
    X=zeros(5,1); x_hat=zeros(5,1); P=eye(5);
    Q=1e-6*eye(5);
    R=1e-2*eye(5);
    L=chol(R,'lower');
    
    innov_store=zeros(5,N);
    gust_store=zeros(3,N);
    
    g_mag=5+10*rand;
    g_freq=0.2+1.0*rand;
    
    for k=1:N
        
        u=[0;0];
        gust=[g_mag*sin(2*pi*g_freq*t(k));
              g_mag*cos(2*pi*g_freq*t(k));
              0];
        
        X = Ad*X + Bd*u + Bwd*gust;
        z = X + L*randn(5,1);
        
        x_pred = Ad*x_hat + Bd*u;
        P_pred = Ad*P*Ad' + Q;
        
        y = z - x_pred;
        
        S = P_pred + R;
        K = P_pred / S;
        
        x_hat = x_pred + K*y;
        P = (eye(5)-K)*P_pred;
        
        innov_store(:,k)=y;
        gust_store(:,k)=gust;
    end
    
    XTrain{s}=innov_store;
    YTrain{s}=gust_store;
end

%% =============================
%  NORMALIZATION
% ==============================

allData = [];
for s=1:numSim
    allData = [allData XTrain{s}];
end

mu = mean(allData,2);
sigma = std(allData,0,2)+1e-6;

for s=1:numSim
    XTrain{s}=(XTrain{s}-mu)./sigma;
end

%% =============================
%  GRU PARAMETERS
% ==============================

inputSize=5;
hiddenSize=32;
outputSize=3;

rng(0)

net.Wz=randn(hiddenSize,inputSize)*0.1;
net.Uz=randn(hiddenSize,hiddenSize)*0.1;
net.bz=zeros(hiddenSize,1);

net.Wr=randn(hiddenSize,inputSize)*0.1;
net.Ur=randn(hiddenSize,hiddenSize)*0.1;
net.br=zeros(hiddenSize,1);

net.Wh=randn(hiddenSize,inputSize)*0.1;
net.Uh=randn(hiddenSize,hiddenSize)*0.1;
net.bh=zeros(hiddenSize,1);

net.Wo=randn(outputSize,hiddenSize)*0.1;
net.bo=zeros(outputSize,1);

%% =============================
%  TRAINING (FULL BPTT)
% ==============================

epochs=25;
lr=0.001;

for epoch=1:epochs
    
    totalLoss=0;
    
    for s=1:numSim
        
        Xseq=XTrain{s};
        Gseq=YTrain{s};
        
        H=zeros(hiddenSize,N+1);
        Z=zeros(hiddenSize,N);
        Rg=zeros(hiddenSize,N);
        Htilde=zeros(hiddenSize,N);
        Ghat=zeros(outputSize,N);
        
        % Forward
        for k=1:N
            x=Xseq(:,k);
            hprev=H(:,k);
            
            z=sig(net.Wz*x+net.Uz*hprev+net.bz);
            r=sig(net.Wr*x+net.Ur*hprev+net.br);
            htil=tanh(net.Wh*x+net.Uh*(r.*hprev)+net.bh);
            
            h=(1-z).*hprev+z.*htil;
            
            H(:,k+1)=h;
            Z(:,k)=z;
            Rg(:,k)=r;
            Htilde(:,k)=htil;
            Ghat(:,k)=net.Wo*h+net.bo;
        end
        
        % Initialize gradients
        grads = init_grads(net);
        dh_next=zeros(hiddenSize,1);
        
        % Backward
        for k=N:-1:1
            
            dy = 2*(Ghat(:,k)-Gseq(:,k));
            totalLoss=totalLoss+sum(dy.^2);
            
            grads.dWo = grads.dWo + dy*H(:,k+1)';
            grads.dbo = grads.dbo + dy;
            
            dh = net.Wo'*dy + dh_next;
            
            z=Z(:,k); r=Rg(:,k);
            htil=Htilde(:,k);
            hprev=H(:,k);
            
            dh_til = dh.*z.*(1-htil.^2);
            dz = dh.*(htil-hprev).*z.*(1-z);
            dr = (net.Uh'*dh_til).*hprev.*r.*(1-r);
            
            grads.dWh = grads.dWh + dh_til*Xseq(:,k)';
            grads.dUh = grads.dUh + dh_til*(r.*hprev)';
            grads.dWz = grads.dWz + dz*Xseq(:,k)';
            grads.dUz = grads.dUz + dz*hprev';
            grads.dWr = grads.dWr + dr*Xseq(:,k)';
            grads.dUr = grads.dUr + dr*hprev';
            
            dh_next = dh.*(1-z) ...
                      + net.Uz'*dz ...
                      + net.Ur'*dr ...
                      + (net.Uh'*dh_til).*r;
        end
        
        % Gradient clipping
        grads = clip_grads(grads,1);
        
        % Update
        net = apply_update(net,grads,lr);
        
    end
    
    fprintf('Epoch %d Loss %.4f\n',epoch,totalLoss/numSim);
end

%% =============================
%  TEST WITH KF + GRU
% ==============================

X=zeros(5,1); x_hat=zeros(5,1); P=eye(5);
Q=1e-6*eye(5); R=1e-2*eye(5);
L=chol(R,'lower');

h=zeros(hiddenSize,1);

gust_est=zeros(3,N);
gust_true=zeros(3,N);

for k=1:N
    
    true_g=[8*sin(2*pi*0.5*t(k));5;0];
    u=[0;0];
    
    X = Ad*X + Bd*u + Bwd*true_g;
    z = X + L*randn(5,1);
    
    x_pred = Ad*x_hat + Bd*u;
    P_pred = Ad*P*Ad' + Q;
    
    y=z-x_pred;
    
    x_input=(y-mu)./sigma;
    
    [g_est,h]=gru_forward(x_input,h,net);
    
    x_pred = x_pred + Bwd*g_est;
    
    S=P_pred+R;
    K=P_pred/S;
    
    x_hat=x_pred+K*(z-x_pred);
    P=(eye(5)-K)*P_pred;
    
    gust_est(:,k)=g_est;
    gust_true(:,k)=true_g;
end

figure
plot(t,gust_true(1,:),t,gust_est(1,:),'--')
legend('True','Estimated')
title('Gust Estimation')

%% =============================
%  FUNCTIONS
% ==============================

function y=sig(x)
y=1./(1+exp(-x));
end

function [g,h]=gru_forward(x,hprev,net)
z=1./(1+exp(-(net.Wz*x+net.Uz*hprev+net.bz)));
r=1./(1+exp(-(net.Wr*x+net.Ur*hprev+net.br)));
htil=tanh(net.Wh*x+net.Uh*(r.*hprev)+net.bh);
h=(1-z).*hprev+z.*htil;
g=net.Wo*h+net.bo;
end

function grads=init_grads(net)
fields=fieldnames(net);
for i=1:length(fields)
    if startsWith(fields{i},'W')||startsWith(fields{i},'U')||startsWith(fields{i},'b')
        grads.(['d' fields{i}])=zeros(size(net.(fields{i})));
    end
end
end

function grads=clip_grads(grads,clip)
fields=fieldnames(grads);
for i=1:length(fields)
    grads.(fields{i})=max(min(grads.(fields{i}),clip),-clip);
end
end

function net=apply_update(net,grads,lr)
fields=fieldnames(grads);
for i=1:length(fields)
    name=fields{i}(2:end);
    net.(name)=net.(name)-lr*grads.(fields{i});
end
end
