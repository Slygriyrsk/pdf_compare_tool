th_kf = 0.3;   % threshold

idx_kf = find(abs(residual) > th_kf, 1);

if ~isempty(idx_kf)
    det_time_kf = t(idx_kf);
    delay_kf = det_time_kf - fault_times(1);
else
    det_time_kf = NaN;
    delay_kf = NaN;
end


figure;
subplot(2,1,1)
plot(t,signals_fault(1,:),'b'); hold on;
plot(t,kf_est,'r--');
xline(fault_times(1),'k--');
title('Signal vs Kalman Estimate');
legend('Measured','Kalman');

subplot(2,1,2)
plot(t,residual,'k');
yline(th_kf,'r--');
xline(fault_times(1),'k--');
title('Kalman Residual (Fault Detection)');
xlabel('Time (s)');
grid on;



scaled = fix((signals_fault(1:4,:) + 50));
packed = uint32(zeros(1,N));

for k = 1:N
    packed(k) = uint32( ...
        scaled(1,k) + ...
        scaled(2,k)*100 + ...
        scaled(3,k)*10000 + ...
        scaled(4,k)*1000000 );
end



rec = zeros(4,N);

for k = 1:N
    rec(1,k) = mod(packed(k),100);
    rec(2,k) = mod(floor(packed(k)/100),100);
    rec(3,k) = mod(floor(packed(k)/10000),100);
    rec(4,k) = mod(floor(packed(k)/1000000),100);
end


diff_rec = [zeros(4,1) diff(rec,1,2)];
th_pack = 3;

for i = 1:4
    idx = find(abs(diff_rec(i,:)) > th_pack,1);
    if ~isempty(idx)
        fprintf('Packed anomaly detected in S%d at %.2f sec\n', ...
                i, t(idx));
    end
end
