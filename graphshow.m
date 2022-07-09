% gurobiじゃないです

% 図を表示
close all
%%{
figure % EP-L-Dualの変化
plot(EP_L_Dual_rec,'LineWidth',1.0); hold on
plot(EP_L_Dual_real_rec,'LineWidth',1.0);
yline(EP,'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
ylim([5000 1.1*EP])
grid on
title('EP-L-Dualの推移')
xlabel('Day'); ylabel('EP-L-Dual')
lgd = legend({'EP-L-Dual','実現値','EP'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

%%{
figure % p(1,1))
p_1_1_rec = zeros(1,iteration);
for i=1:iteration
    p_1_1_rec(1,i) = p_rec(1,1,i);
end
plot(p_1_1_rec,'LineWidth',1.0); hold on
yline(EP_p(1,1),'--r','LineWidth',0.8); hold off
grid on
title('p(1,1)の変化')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');

figure % p(1,2))
plot(p_1_2_rec,'LineWidth',1.0); hold on
yline(EP_p(1,2),'--r','LineWidth',0.8); hold off
grid on
title('p(1,2)の変化')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');

%%{
figure % p(5,3)
p_5_3_rec = zeros(1,iteration);
for i=1:iteration
    p_5_3_rec(1,i) = p_rec(5,3,i);
end
plot(p_5_3_rec,'LineWidth',1.0); hold on
yline(EP_p(5,3),'--r','LineWidth',0.8); hold off
grid on
title('p(5,3)の変化')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

%%{
figure % p(5,4)
p_5_4_rec = zeros(1,iteration);
for i=1:iteration
    p_5_4_rec(1,i) = p_rec(5,4,i);
end
plot(p_5_4_rec,'LineWidth',1.0); hold on
yline(EP_p(5,4),'--r','LineWidth',0.8); hold off
grid on
title('p(5,4)の変化')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

%%{
figure % p(6,4)
p_6_4_rec = zeros(1,iteration);
for i=1:iteration
    p_6_4_rec(1,i) = p_rec(6,4,i);
end
plot(p_6_4_rec,'LineWidth',1.0); hold on
yline(EP_p(6,4),'--r','LineWidth',0.8); hold off
grid on
title('ij=6,t=4のサービス価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%}

figure % p(1,7)
plot(p_1_7_rec,'LineWidth',1.0); hold on
yline(EP_p(1,7),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=1,t=7)のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');

%{
figure % p(2,7)
plot(p_2_7_rec,'LineWidth',1.0); hold on
yline(EP_p(2,7),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('ij=2,t=7のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%}

%{
figure % p(1,9)
plot(p_1_9_rec,'LineWidth',1.0); hold on
yline(EP_p(1,9),'--r','LineWidth',0.8); hold off
grid on
title('p(1,9)の変化')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%}

%%{
figure % p(5,9)
plot(p_5_9_rec,'LineWidth',1.0); hold on
yline(EP_p(5,9),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=5,t=9)のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

%{
figure % p(7,10)
plot(p_7_10_record,'LineWidth',1.0); hold on
yline(EP_p(7,10),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('ij=7,t_{ij}=10のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%}

%{
figure %経路1全体のMS価格の推移
plot(p_1_7_record+p_5_9_record,'LineWidth',1.0); hold on
yline(EP_p(1,7)+EP_p(5,9),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('経路全体のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'総MS価格','均衡総MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%}

%{
figure %経路2全体のMS価格の推移
plot(p_2_7_record+p_7_10_record,'LineWidth',1.0); hold on
yline(EP_p(2,7)+EP_p(7,10),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('経路2全体のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'総MS価格','均衡総MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%}

%{
figure % p(1,10))
plot(p_1_10_record,'LineWidth',1.0); hold on
yline(EP_p(1,10),'--r','LineWidth',0.8); hold off
grid on
title('p(1,10)の変化')
xlabel('Day'); ylabel('価格')

figure % p(1,11))
plot(p_1_11_record,'LineWidth',1.0); hold on
yline(EP_p(1,11),'--r'); hold off
grid on
title('p(1,11)の変化','LineWidth',0.8)
xlabel('Day'); ylabel('価格')
%}

%{
figure % p(5,11)
plot(p_5_11_record,'LineWidth',1.0); hold on
yline(EP_p(5,11),'--r','LineWidth',0.8); hold off
grid on
title('p(5,11)の変化')
xlabel('Day'); ylabel('価格')
%}

%{
figure % p(5,12))
plot(p_5_12_record,'LineWidth',1.0); hold on
yline(EP_p(5,12),'--r','LineWidth',0.8); hold off
grid on
title('p(5,12)の変化')
xlabel('Day'); ylabel('価格')
%}

%{
figure % p(5,13)
plot(p_5_13_record,'LineWidth',1.0); hold on
yline(EP_p(5,13),'--r','LineWidth',0.8); hold off
grid on
title('p(5,13)の変化')
xlabel('Day'); ylabel('価格')
%}
%}

figure % 全リンクの総超過需要の推移
plot(link_dem_excess_sum_rec,'LineWidth',1.0)
xlim([-0.1*s 1.1*s])
grid on
title('全リンクの総超過需要の推移')
xlabel('Day'); ylabel('総超過需要の合計')
lgd = legend({'全リンクの総超過需要'},'FontSize',14,'TextColor','black','Location','northeast');

figure % Z_ave
plot(Z_ave_rec,'LineWidth',1.0); hold on
yline(0,'--r','LineWidth',0.8); hold off
grid on
title('市場精算過程')
xlabel('Day'); ylabel('')

acolor = zeros(SP_U_X_num,3);
for i=1:SP_U_X_num
    acolor(i,:) = [rand rand rand];
end

figure % 価格調整中に発生するOD経路別の費用
plot(OD_each(1,:),'LineWidth',0.5,'Color',acolor(1,:)); hold on
for i=2:SP_U_X_num
    plot(OD_each(i,:),'LineWidth',0.5,'Color',acolor(i,:));
end
%yline(EP_SPU,'--r','LineWidth',0.8); hold off
grid on
title('[EP-L-Dual]で発生する各最適OD経路別の最小費用')
xlabel('Day'); ylabel('費用')

%{
figure % EP-L-Dual
plot(link_EPp_p_abs_sum_record,'LineWidth',1.0); hold on
%yline(EP,'--r','LineWidth',0.8); hold off
grid on
title('探索価格と均衡価格の差の絶対値の合計の変化')
xlabel('Day'); ylabel('探索価格と均衡価格の差の絶対値の合計')

figure % EP-L-Dual
plot(link_EPp_p_sum_record,'LineWidth',1.0); hold on
%yline(EP,'--r','LineWidth',0.8); hold off
grid on
title('探索価格と均衡価格の差の合計の変化')
xlabel('Day'); ylabel('探索価格と均衡価格の差の合計')

figure % EP-L-Dual
plot(link_EPp_p_mean_record,'LineWidth',1.0); hold on
%yline(EP,'--r','LineWidth',0.8); hold off
grid on
title('探索価格と均衡価格の相対誤算の平均値')
xlabel('Day'); ylabel('探索価格と均衡価格の相対誤算の平均値')
%}