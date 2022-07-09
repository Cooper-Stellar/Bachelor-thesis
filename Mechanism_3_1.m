clear all

iteration = 5000; % 反復回数

% 与件のパラメータ
T = 25; % 時間ステップ数
alfa = 30; % 移動時間価値(円/min)
eps = 1000; % 収束判定値

% ネットワーク情報(リンク番号，発ノード，着ノード，自由旅行時間，容量)
road = [
    1 1 2 2 80
    2 1 3 3 30
    3 2 1 2 80
    4 2 3 2 40
    5 2 4 3 40
    6 3 1 3 30
    7 3 4 3 70
    8 4 2 3 40
    9 4 3 3 70
    %10 3 2 2 40
    ];
N = max([max(road(:,2)) max(road(:,3))]); % ノード数
L = height(road); % リンク数
t = repmat(road(:,4),1,T-1); % (L,(T-1))に対応した自由旅行時間
limit = repmat(road(:,5),1,T-1); % (L,(T-1))に対応した道路容量

% OD(利用者)情報(出発ノード，到着ノード，希望到着時間，利用者数)
user = [ % routeをuserになおす（確認）
    1 4 12 60
    %1 4 12 1600
    %3 2 12 1600
    %4 1 12 1600
    ];
OD = height(user); % ODトリップ数
O_node = user(:,1)'; % 各経路の出発ノード
D_node = user(:,2)'; % 各経路の到着ノード
D_hope = user(:,3)'; % 各経路の希望到着時間
Q = user(:,4)'; % 各経路の利用者数

% SAVプロバイダー情報(SAV番号，拠点，保有SAV車両数，車両容量，サービス費用)
%{
SAV = [
    1 1 60 3 100
    %2 3 60 3 100
    ];
%}
%%{
SAV = [
    1 1 20
    %2 3 500
    ];
%%}
R = height(SAV); % SAVプロバイダーの数
R_base = SAV(:,2)'; % 拠点
S = SAV(:,3)'; % 1社が保有するSAV車両数
C = 4; % 車両容量(人/台)
d = 20.*t;
%d = 20.*(t+rand(L,T-1).*0.01); % サービス費用(円)(今回は全てのリンクで同じ)
beta = 50; % アクティブ車両数１台あたりのコスト係数(円/台)
p0 = zeros(L,T-1,R); % MS価格の初期値
%{
for i=1:R
    %p0(:,:,i) = rand(L,T-1).*100;
    %p0(:,:,i) = ones(L,T-1).*100;
    p0(:,:,i) = zeros(L,T-1).*100;
end
%}
p = p0;

% スケジュール費用
w = zeros(1,T*OD);
for i=1:OD
    for j=1:D_hope(1,i)
        w(1,T*(i-1)+j) = 1*(D_hope(1,i)-j)*25;
    end
    for j=D_hope(1,i)+1:T
        w(1,T*(i-1)+j) = 2*(j-D_hope(1,i))*25;
    end
end


%% 変数の準備
% 目的変数の係数
t_ij = repmat(reshape(t,1,[]),1,OD);
x_co = ones(1,L*(T-1)*OD).*alfa.*t_ij;
f_co = ones(1,T*OD).*w;
d_ij = repmat(reshape(d,1,[]),1,R);
y_co = ones(1,L*(T-1)*R).*d_ij;
g_co = zeros(1,T*R);
h_co = ones(1,T*R).*beta;
coefficient = [x_co f_co y_co g_co h_co]; % 係数のlinprog
column_long = L*(T-1)*(OD+R)+T*(OD+R*2);


% 各主体のノードのフロー保存則のフローブロックを作成
flow_block = zeros(N*T,L*(T-1));
for i=1:L %リンク
    j=1; %時間
    while N*(j-1+road(i,4))+road(i,3)<=N*T
        flow_block(N*(j-1)+road(i,2),L*(j-1)+i) = 1; % 流出
        flow_block(N*(j-1+road(i,4))+road(i,3),L*(j-1)+i) = -1; % 流入
        j=j+1;
    end
end


% 時空間ネットワークの関係ないリンクの容量を0にする(Aeq)
useless_link = zeros(1,L*(T-1));
for i=1:L
    if road(i,4)>1
        for j=2:road(i,4)
            useless_link(1,L*(T-j)+i) = 1;
        end
    end
end
useless_block = [repmat(useless_link,1,OD) zeros(1,T*OD) repmat(useless_link,1,R) zeros(1,T*R*2)];


% 利用者のノードのフロー保存則
node_flow_user = zeros((N-1)*T*OD,column_long);
user_flow_block = zeros(N*T*OD,L*(T-1)*OD);
for i=1:OD
    user_flow_block(N*T*(i-1)+1:N*T*i,L*(T-1)*(i-1)+1:L*(T-1)*i) = flow_block;
end
for i=1:OD
    %i
    num=0;
    for j=1:N*T
        if mod(j,N) ~= mod(O_node(1,i),N)
            num=num+1;
            node_flow_user((N-1)*T*(i-1)+num,1:L*(T-1)*OD) = user_flow_block(N*T*(i-1)+j,:);
        end
    end
    if O_node(1,i)<D_node(1,i)
        D_node(1,i)=D_node(1,i)-1;
    end
    for j=1:T
        node_flow_user((N-1)*T*(i-1)+(N-1)*(j-1)+D_node(1,i),L*(T-1)*OD+T*(i-1)+j)=1; % fのところ
    end
end


% 利用者のODフロー保存則
OD_flow_user = zeros(OD,column_long);
for i=1:OD
    OD_flow_user(i,L*(T-1)*OD+T*(i-1)+1:L*(T-1)*OD+T*(i-1)+T) = ones(1,T)*(-1);
end


% SAV車両のノードのフロー保存則
node_flow_sav = zeros(N*T*R,column_long);
for i=1:R
    node_flow_sav(N*T*(i-1)+1:N*T*i,L*(T-1)*OD+T*OD+L*(T-1)*(i-1)+1:L*(T-1)*OD+T*OD+L*(T-1)*i) = flow_block;
    for j=1:T
        node_flow_sav(N*T*(i-1)+N*(j-1)+R_base(1,i),L*(T-1)*(OD+R)+T*OD+T*(i-1)+j)=1; % gのところ
    end
end
node_flow_sav = node_flow_sav*(-1);


% SAV車両のODフロー保存則(hとgであらわされるもの)(Aeq,beq)
OD_flow_sav = zeros(T*R,column_long);
for i=1:T
    for j=0:R-1
        OD_flow_sav(T*j+i,L*(T-1)*(OD+R)+T*OD+T*j+1:L*(T-1)*(OD+R)+T*OD+T*j+i) = ones(1,i); % gのところ
        OD_flow_sav(T*j+i,L*(T-1)*(OD+R)+T*(OD+R)+T*j+i) = 1; % hのところ
    end
end


% 時刻TでSAV車両は0
sav_end = zeros(R,column_long);
for i=R-1:-1:0
    sav_end((i-R)*(-1),column_long-T*i) = 1;
end


% 道路容量制約(A,b)
road_matrix = zeros(L*(T-1),column_long);
for i=1:T-1
    for j=1:L
        for k=0:R-1
            road_matrix(L*(i-1)+j,L*(T-1)*OD+T*OD+L*(T-1)*k+L*(i-1)+j) = 1; % yのところ
        end
    end
end


% MSサービス供給制約(A,b)
service_matrix = zeros(L*(T-1),column_long);
for i=1:T-1
    for j=1:L
        for k=0:OD-1
            service_matrix(L*(i-1)+j,L*(T-1)*k+L*(i-1)+j) = 1; % xのところ
        end
        for k=0:R-1
            service_matrix(L*(i-1)+j,L*(T-1)*OD+T*OD+L*(T-1)*k+L*(i-1)+j) = C*(-1); % yのところ
        end
    end
end



%% linprogを回す
A = [road_matrix; service_matrix];
%A_S = sparse(A);
b = [ones(1,L*(T-1)).*reshape(limit,1,[]) zeros(1,L*(T-1))];
Aeq = [node_flow_user; OD_flow_user; node_flow_sav; OD_flow_sav; sav_end; useless_block];
%Aeq_S = sparse(Aeq);
beq = [zeros(1,(N-1)*T*OD) ones(1,OD).*Q.*(-1) zeros(1,N*T*R) zeros(1,T*R) zeros(1,R) 0];
lb_SAVnet = ones(1,T*R);
for i=1:R
    lb_SAVnet(1,T*(i-1)+1:T*i) = S(1,i);
end
%lb = sparse([zeros(1,(L*(T-1)*(OD+R)+T*OD)) lb_SAVnet.*(-1) zeros(1,T*R)]);
lb = [zeros(1,(L*(T-1)*(OD+R)+T*OD)) -Inf(1,T*R) zeros(1,T*R)];
%{
ub_ODflow = ones(1,T*OD);
for i=1:OD
    ub_ODflow(1,T*(i-1)+1:T*i) = Q(1,i);
end
ub = sparse([C.*repmat(reshape(limit,1,[]),1,OD) ub_ODflow repmat(reshape(limit,1,[]),1,R) lb_SAVnet lb_SAVnet]);
%}
ub = [Inf(1,L*(T-1)*(OD+R)+T*(OD+R)) lb_SAVnet];

options = optimoptions('linprog','Algorithm','dual-simplex');
%options = optimoptions('linprog','Algorithm','interior-point-legacy');
%options = optimoptions('linprog','Algorithm','interior-point');
[EPX,EP,exitflag,output,lambda] = linprog(coefficient, A, b, Aeq, beq, lb, ub, options);
%[EPX,EP,exitflag,output,lambda] = linprog(coefficient, A_S, b, Aeq_S, beq, lb, ub, options);



% 均衡状態の数値
EP_x = reshape(EPX(1:L*(T-1)*OD,1),L,(T-1)*OD);
EP_x_3dim = zeros(L,T-1,OD);
EP_x_sum = zeros(L,T-1);
for i=1:OD
    EP_x_3dim(:,:,i) = EP_x(:,(T-1)*(i-1)+1:(T-1)*i);
    EP_x_sum = EP_x_sum + EP_x_3dim(:,:,i);
end
if OD==1
    EP_x;
elseif OD==2
    EP_x1 = EP_x(:,1:T-1);
    EP_x2 = EP_x(:,(T-1)+1:(T-1)*2);
elseif OD==3
    EP_x1 = EP_x(:,1:T-1);
    EP_x2 = EP_x(:,(T-1)+1:(T-1)*2);
    EP_x3 = EP_x(:,(T-1)*2+1:(T-1)*3);
elseif OD==4
    EP_x1 = EP_x(:,1:T-1);
    EP_x2 = EP_x(:,(T-1)+1:(T-1)*2);
    EP_x3 = EP_x(:,(T-1)*2+1:(T-1)*3);
    EP_x4 = EP_x(:,(T-1)*3+1:(T-1)*4);
end
EP_f = reshape(EPX(L*(T-1)*OD+1:L*(T-1)*OD+T*OD,1),T,OD)';
EP_y = reshape(EPX(L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*OD,1),L,(T-1)*R);
EP_y_3dim = zeros(L,T-1,R);
EP_y_sum = zeros(L,T-1);
for i=1:R
    EP_y_3dim(:,:,i) = EP_y(:,(T-1)*(i-1)+1:(T-1)*i);
    EP_y_sum = EP_y_sum + EP_y_3dim(:,:,i);
end
%EP_y_exist = (EP_y_sum > 0.001);
if R==1
    EP_y1 = EP_y(:,1:T-1);
elseif R==2
    EP_y1 = EP_y(:,1:T-1);
    EP_y2 = EP_y(:,(T-1)+1:(T-1)*2);
elseif R==3
    EP_y1 = EP_y(:,1:T-1);
    EP_y2 = EP_y(:,(T-1)+1:(T-1)*2);
    EP_y3 = EP_y(:,(T-1)*2+1:(T-1)*3);
elseif R==4
    EP_y1 = EP_y(:,1:T-1)
    EP_y2 = EP_y(:,(T-1)+1:(T-1)*2)
    EP_y3 = EP_y(:,(T-1)*2+1:(T-1)*3)
    EP_y4 = EP_y(:,(T-1)*3+1:(T-1)*4);
end
EP_g = reshape(EPX(L*(T-1)*(OD+R)+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R),1),T,R)';
EP_h = reshape(EPX(L*(T-1)*(OD+R)+T*(OD+R)+1:L*(T-1)*(OD+R)+T*(OD+R*2),1),T,R)';
EP_SPU = (x_co+(lambda.ineqlin(L*(T-1)+1:L*(T-1)*2,1))')*EPX(1:L*(T-1)*OD,1) + f_co*EPX(L*(T-1)*OD+1:L*(T-1)*OD+T*OD,1);


% ラグランジュ乗数の生成
EP_e = reshape(lambda.ineqlin(1:L*(T-1),1),L,(T-1));
EP_p = reshape(lambda.ineqlin(L*(T-1)+1:L*(T-1)*2,1),L,(T-1));
%EP_p = EP_p.*EP_y_exist;
EP_p_round = round(EP_p);
EP_u = reshape(lambda.eqlin(1:(N-1)*T*OD,1),N-1,T*OD);
if OD==1
    EP_u1 = EP_u(:,1:T);
elseif OD==2
    EP_u1 = EP_u(:,1:T);
    EP_u2 = EP_u(:,T+1:T*2);
elseif OD==3
    EP_u1 = EP_u(:,1:T);
    EP_u2 = EP_u(:,T+1:T*2);
    EP_u3 = EP_u(:,T*2+1:T*3);
end
EP_rho = lambda.eqlin((N-1)*T*OD+1:(N-1)*T*OD+OD,1);
EP_v = reshape(lambda.eqlin((N-1)*T*OD+OD+1:(N-1)*T*OD+OD+N*T*R,1),N,T*R);
if R==1
    EP_v1 = EP_v(:,1:T);
elseif R==2
    EP_v1 = EP_v(:,1:T);
    EP_v2 = EP_v(:,T+1:T*2);
elseif R==3
    EP_v1 = EP_v(:,1:T);
    EP_v2 = EP_v(:,T+1:T*2);
    EP_v3 = EP_v(:,T*2+1:T*3);
end
EP_phi = reshape(lambda.eqlin((N-1)*T*OD+OD+N*T*R+1:(N-1)*T*OD+OD+N*T*R+T*R,1),T,R)';
EP_v_dif = (-1)*EP_phi;
EP_pi = reshape(lambda.upper((T-1)*L*(OD+R)+T*(OD+R)+1:(T-1)*L*(OD+R)+T*(OD+R*2),1),T,R)';



%% 到達メカニズム
s=0; % 日数
%iteration = 5000; % 反復回数
eps = EP / 100; % 収束判定値
x = zeros(L,T-1,OD);
f = zeros(1,T,OD);
y = zeros(L,T-1,R);
h = zeros(1,T,R);
EP_L_Dual = 0; % EP-L-Dual
x_sum_rec = zeros(L,T-1,iteration);
y_sum_rec = zeros(L,T-1,iteration);
Z_dif = zeros(L,T-1,R); %現在の超過需要
Z_total = zeros(L,T-1,R); %累計の超過需要
link_demsup_excess_rec = zeros(L,T-1,iteration);
link_demsup_excess_ave = zeros(L,T-1,iteration);
link_demsup_excess_avesum = zeros(L,T-1,iteration);
link_dem_excess_rec = zeros(L,T-1,iteration);
link_sup_excess_rec = zeros(L,T-1,iteration);


% グラフ用
Z_total_rec = zeros(L,T-1,iteration);
EP_L_Dual_rec = zeros(1,iteration);
EP_L_Dual_real_rec = zeros(1,iteration);
link_dem_excess_sum_rec = zeros(1,iteration);
link_sup_excess_sum_rec = zeros(1,iteration);
link_EPp_p_abs_sum_record = zeros(1,iteration);
link_EPp_p_sum_record = zeros(1,iteration);
link_EPp_p_mean_record = zeros(1,iteration);
Z_ave_rec = zeros(1,iteration);
SP_U_X_rec = zeros(iteration,L*(T-1)*OD+T*OD);
xf_co_rec = zeros(iteration,L*(T-1)*OD+T*OD);


%収束のパーセンテージ
Eq_90 = zeros(1,2);
Eq_95 = zeros(1,2);
Eq_99 = zeros(1,2);
Eq_99_99 = zeros(1,2);

% 記録
p_rec = zeros(L,T-1,(iteration));

%{
p0 = ones(L,T-1,R); % MS価格の初期値
for i=1:R
    %p0(:,:,i) = rand(L,T-1).*100;
    p0(:,:,i) = EP_p;
end
p = p0;
%}


% 図の作成
%{
clear vec1
clear vec2
clear vec3
clear vec4
figure
hold on;
xlim([0 iteration])
%ylim([0 600000])
vec1(1) = 0;
vec2(1) = 0;
%vec3(0) = 0;
%vec4(0) = 0;
h1 = plot(vec1,vec2, 'o-');
xlabel('Day')
%ylabel('合計超過需要数')
%h2 = plot(vec1,vec3, 'o-');
%xlabel('Day'); ylabel('リンク利用者(ij=5,t=13)')
%h3 = plot(vec1,vec4, 'o-');
%xlabel('Day'); ylabel('リンクSAV車両(ij=5,t=13)')
%}

options = optimoptions('linprog','Display','none'); %「最適解が見つかりました」が出ないようにする
tic
%while abs(EP-EP_L_Dual)>eps
while s<iteration
    % 記録
    s=s+1;
    %p_record(:,:,s) = p;
    %{
    for i=1:R
        p_record(:,:,i,s) = p(:,:,i);
    end
    %}
   
    
    
    %% [SP-P]
    % [SP-P]目的変数の係数
    d_ij = repmat(reshape(d,1,[]),1,R);
    y_co = ones(1,L*(T-1)*R).*(d_ij-C.*reshape(p,1,[]));
    g_co = zeros(1,T*R);
    h_co = ones(1,T*R).*beta;
    coefficient_P = sparse([y_co g_co h_co]); % 係数のlinprog
    column_long_P = L*(T-1)*R+T*R*2;
    
    % [SP-P]linprogを回す
    A = road_matrix(:,L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R*2));
    %A_S = sparse(A);
    b = ones(1,L*(T-1)).*reshape(limit,1,[]);
    Aeq = [
        node_flow_sav(:,L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R*2));
        OD_flow_sav(:,L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R*2));
        sav_end(:,L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R*2))
        useless_block(:,L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R*2))
        ];
    %Aeq_S = sparse(Aeq);
    beq = sparse([zeros(1,N*T*R) zeros(1,T*R) zeros(1,R) 0]);
    %lb = sparse([zeros(1,(L*(T-1)*R)) lb_SAVnet.*(-1) zeros(1,T*R)]);
    lb = [zeros(1,(L*(T-1)*R)) -Inf(1,T*R) zeros(1,T*R)];
    %ub = sparse([repmat(reshape(limit,1,[]),1,R) lb_SAVnet lb_SAVnet]);
    ub = [Inf(1,L*(T-1)*R+T*R) lb_SAVnet];
    [SP_P_Y,SP_P] = linprog(coefficient_P, A, b, Aeq, beq, lb, ub, options);
    SP_P;
    y_new = reshape(SP_P_Y(1:L*(T-1)*R,1),L,T-1,R);
    h_new = reshape(SP_P_Y(L*(T-1)*R+T*R+1:L*(T-1)*R+T*R*2,1),1,T,R);
    y = ((s-1)/s)*y+(1/s)*y_new;
    h = ((s-1)/s)*h+(1/s)*h_new;
    
    
    %% [SP-U]
    % 最小価格を特定，需要シェアを決定
    %%{
    p_min = zeros(L,(T-1));
    demand_share = zeros(L,(T-1),R); %ODによらない，等分
    for j=1:T-1
        for i=1:L
            p_min(i,j) = min(p(i,j,:));
            num = 0;
            for k=1:R
                if p(i,j,k)==p_min(i,j)
                    demand_share(i,j,k) = 1;
                    num = num+1;
                end
            end
            demand_share(i,j,:) = demand_share(i,j,:)/num;
        end
    end
    %%}
    %{
    p_round = round(p); %整数化
    p_min = zeros(L,(T-1));
    demand_share = zeros(L,(T-1),R); %ODによらない，等分
    for j=1:T-1
        for i=1:L
            p_min(i,j) = min(p_round(i,j,:));
            num = 0;
            for k=1:R
                if p_round(i,j,k)==p_min(i,j)
                    demand_share(i,j,k) = 1;
                    num = num+1;
                end
            end
            demand_share(i,j,:) = demand_share(i,j,:)/num;
        end
    end
    %}
    p_rec(:,:,s) = p_min;
    
    
    % [SP-U]目的変数の係数
    t_ij = repmat(reshape(t,1,[]),1,OD);
    x_co = ones(1,L*(T-1)*OD).*(alfa.*t_ij+repmat(reshape(p_min,1,[]),1,OD));
    f_co = ones(1,T*OD).*w;
    coefficient_U = sparse([x_co f_co]);
    column_long_U = L*(T-1)*OD+T*OD;
    
    xf_co_rec(s,:) = [x_co f_co];
    
    % [SP-U]linprogを回す
    A = [];
    b = [];
    %A = [service_matrix(:,1:L*(T-1)*OD+T*OD)];
    %b = [C*ones(1,L*(T-1)).*reshape(limit,1,[])];
    Aeq = [
        node_flow_user(:,1:L*(T-1)*OD+T*OD);
        OD_flow_user(:,1:L*(T-1)*OD+T*OD);
        useless_block(:,1:L*(T-1)*OD+T*OD);
        ];
    %Aeq_S = sparse(Aeq);
    beq = sparse([zeros(1,(N-1)*T*OD) ones(1,OD).*Q.*(-1) 0]);
    %lb = sparse([zeros(1,(L*(T-1)*OD+T*OD))]);
    lb = zeros(1,(L*(T-1)*OD+T*OD));
    %ub = sparse([inf(1,L*(T-1)*OD) ub_ODflow]);
    %ub = [C.*repmat(reshape(limit,1,[]),1,OD) ub_ODflow];
    ub = inf(1,L*(T-1)*OD+T*OD);
    [SP_U_X,SP_U] = linprog(coefficient_U, A, b, Aeq, beq, lb, ub, options);
    SP_U_X_rec(s,:) = SP_U_X;
    x_new = reshape(SP_U_X(1:L*(T-1)*OD,1),L,T-1,OD);
    f_new = reshape(SP_U_X(L*(T-1)*OD+1:L*(T-1)*OD+T*OD,1),1,T,OD);
    x = ((s-1)/s)*x+(1/s)*x_new;
    f = ((s-1)/s)*f+(1/s)*f_new;
    
    
    
    %% いろいろ
    % EP-L-Dualと実現値(real_value)を計算
    EP_L_Dual = SP_P+SP_U;
    EP_L_Dual_rec(1,s) = EP_L_Dual;
    EP_L_Dual_real = sum(x_co.*reshape(x,1,[])) + sum(f_co.*reshape(f,1,[])) + sum(y_co.*reshape(y,1,[])) + sum(h_co.*reshape(h,1,[]));
    EP_L_Dual_real_rec(1,s) = EP_L_Dual_real;
    if EP_L_Dual/EP>0.9 && Eq_90(1,1)==0
        Eq_90(1,1) = 1;
        Eq_90(1,2) = s;
    end
    if EP_L_Dual/EP>0.95 && Eq_95(1,1)==0
        Eq_95(1,1) = 1;
        Eq_95(1,2) = s;
    end
    if EP_L_Dual/EP>0.99 && Eq_99(1,1)==0
        Eq_99(1,1) = 1;
        Eq_99(1,2) = s;
    end
    if EP_L_Dual/EP>0.9999 && Eq_99_99(1,1)==0
        Eq_99_99(1,1) = 1;
        Eq_99_99(1,2) = s;
    end
    
    % 諸々の計算
    %total_demand_excess = sum(repmat(reshape(limit,1,[]),1,OD).*SP_U_X(1:L*(T-1)*OD,1)' - repmat(reshape(limit,1,[]),1,R).*SP_P_Y(1:L*(T-1)*R,1)'.*C);
    x_sum = zeros(L,T-1);
    for i=1:OD
        x_sum = x_sum + x(:,:,i);
    end
    x_sum_rec(:,:,s) = x_sum;
    y_sum = zeros(L,T-1);
    for i=1:R
        y_sum = y_sum + y(:,:,i);
    end
    y_sum_rec(:,:,s) = y_sum;
    link_demsup_excess_rec(:,:,s) = x_sum - C.*y_sum;
    if s>1
        link_demsup_excess_ave(:,:,s) = (link_demsup_excess_ave(:,:,s-1).*(s-1)+link_demsup_excess_rec(:,:,s))./s;
    end
    link_demsup_excess_avesum = sum(sum(link_demsup_excess_ave(:,:,s).^2));
    
    link_dem_excess_now = link_demsup_excess_rec(:,:,s);
    link_dem_excess_now(link_dem_excess_now < 0) = 0;
    link_dem_excess_rec(:,:,s) = link_dem_excess_now;
    link_dem_excess_sum = sum(sum(link_dem_excess_rec(:,:,s)));
    link_dem_excess_sum_rec(1,s) = link_dem_excess_sum;

    link_sup_excess_now = - link_demsup_excess_rec(:,:,s);
    link_sup_excess_now(link_sup_excess_now < 0) = 0;
    link_sup_excess_rec(:,:,s) = link_sup_excess_now;
    link_sup_excess_sum = sum(sum(link_sup_excess_rec(:,:,s)));
    link_sup_excess_sum_rec(1,s) = link_sup_excess_sum;
    
    %{
    link_EPp_p_abs = abs(EP_p - p_min);
    link_EPp_p_abs_sum = sum(sum(link_EPp_p_abs));
    link_EPp_p_mean = mean(link_EPp_p_abs,'all')/mean(EP_p,'all');  %絶対誤差の平均値の平均価格による規格化
    link_EPp_p_mean_record(1,s) = link_EPp_p_mean;
    link_EPp_p_abs_sum_record(1,s) = link_EPp_p_abs_sum;
    link_EPp_p = EP_p - p_min;
    link_EPp_p_sum = sum(sum(link_EPp_p));
    link_EPp_p_sum_record(1,s) = link_EPp_p_sum;
    %}
    
    
    
    %% 価格の更新
    % [EP-L-Dual]の現在の超過需要を決定
    x_new_sum = zeros(L,T-1);
    for i=1:OD
        x_new_sum = x_new_sum + x_new(:,:,i);
    end
    Z_dif = demand_share.*x_new_sum - C.*y_new; %現在の超過需要
    
    
    % 累計の超過需要を決定
    Z_total = Z_total + Z_dif;
    Z_total_rec(:,:,s) = Z_total;
    Z_ave=sum(sum(Z_total./s.*p))./(L*(T-1));
    Z_ave_rec(1,s) = Z_ave;
    
    
    % 予想価格を計算
    zeta = 5; %スケーリング係数
    p_forecast = zeta.*(1/sqrt(s)).*max(Z_total,0);
    %p_forecast(p_forecast<0)=0;
    
    
    % 価格ベクトルを更新
    p_new = ((s+1)/(s+2)).*p + (1/(s+2)).*p_forecast;
    %p_difference(:,:,s) = p_new - p;
    p = p_new;
    
    
    %2次ペナルティの計算
    %{
    x_sum=zeros(L,T-1);
    y_sum=zeros(L,T-1);
    for i=1:OD
        x_sum = x_sum + x(:,:,i);
    end
    for i=1:R
        y_sum = y_sum + y(:,:,i);
    end
    penalty=sum(sum((x_sum-y_sum).^2));
    %}
    
    
    %{
    xlim([0 s])
    vec1(s) = s;
    %vec2(s) = p(5,13);
    %vec3(s) = x(5,13);
    %vec4(s) = C.*y(5,13);
    %vec2(s) = -SP_P;
    %vec2(s) = SP_U;
    %vec2(s) = penalty; %100000に近づいている
    %vec2(s) = demand_excess;
    %vec2(s) = EP_L_Dual; ylabel('EP-L-Dual')
    %vec2(s) = (EP_L_Dual/EP)*100; ylabel('(EP-L-Dual/EP)*100')
    %vec2(s) = link_dem_excess_sum; ylabel('超過需要の合計')
    %vec2(s) = link_sup_excess_sum; ylabel('超過供給の合計')
    %vec2(s) = max(max(x_sum));
    %vec2(s) = num;
    %vec2(s) = link_demsup_excess_avesum;
    %vec2(s) = link_EPp_p_abs_sum; ylabel('探索価格と均衡価格の差の絶対値の合計')
    vec2(s) = link_EPp_p_mean;
    h1.XData = vec1;
    h1.YData = vec2;
    %h2.XData = vec1;
    %h2.YData = vec3;
    %h3.XData = vec1;
    %h3.YData = vec4;
    drawnow
    pause(0.001)
    %}
end
T1=toc



%% 図の作成
close all

p_figrec = permute(p_rec,[1 3 2]);
x_sum_figrec = permute(x_sum_rec,[1 3 2]);
y_sum_figrec = permute(y_sum_rec,[1 3 2]);

figure % EP-L-Dualと実現値の推移
plot(EP_L_Dual_rec,'LineWidth',1.0); hold on
plot(EP_L_Dual_real_rec,'LineWidth',1.0);
yline(EP,'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
ylim([5000 1.1*EP])
grid on
title('EP-L-Dualと実現値の推移')
xlabel('Day'); ylabel('EP-L-Dual')
lgd = legend({'EP-L-Dual','実現値','EP'},'FontSize',14,'TextColor','black','Location','southeast');

figure %超過需要の合計の変化
plot(link_dem_excess_sum_rec,'LineWidth',0.8);
xlim([-0.1*s 1.1*s])
grid on
title('全リンクの実際の超過需要の合計の推移')
xlabel('Day'); ylabel('総超過需要')
lgd = legend({'総超過需要'},'FontSize',14,'TextColor','black','Location','northeast');

figure % p(1,7)
plot(p_figrec(1,:,7),'LineWidth',1.0); hold on
yline(EP_p(1,7),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=1,t=7)のサービス価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');

%%{
figure % p(5,9)
plot(p_figrec(5,:,9),'LineWidth',1.0); hold on
yline(EP_p(5,9),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=5,t=9)のサービス価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'MS価格','均衡MS価格'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

figure %経路全体のMS価格の推移
plot(p_figrec(1,:,7)+p_figrec(5,:,9),'LineWidth',1.0); hold on
yline(EP_p(1,7)+EP_p(5,9),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('経路全体のMS価格の推移')
xlabel('Day'); ylabel('価格')
lgd = legend({'総MS価格推移','均衡総MS価格'},'FontSize',14,'TextColor','black','Location','southeast');

figure % x_sum(1,7)
plot(x_sum_figrec(1,:,7),'LineWidth',1.0); hold on
yline(EP_x_sum(1,7),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=1,t=7)の利用者数の推移')
xlabel('Day'); ylabel('利用者数')
lgd = legend({'利用者数','均衡利用者数'},'FontSize',14,'TextColor','black','Location','southeast');

%%{
figure % x_sum(5,9)
plot(x_sum_figrec(5,:,9),'LineWidth',1.0); hold on
yline(EP_x_sum(5,9),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=5,t=9)の利用者数の推移')
xlabel('Day'); ylabel('利用者数')
lgd = legend({'利用者数','均衡利用者数'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

figure % y_sum(1,7)
plot(y_sum_figrec(1,:,7),'LineWidth',1.0); hold on
yline(EP_y_sum(1,7),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=1,t=7)のSAV車両数の推移')
xlabel('Day'); ylabel('SAV車両数')
lgd = legend({'SAV車両数','均衡SAV車両数'},'FontSize',14,'TextColor','black','Location','southeast');

%%{
figure % y_sum(5,9)
plot(y_sum_figrec(5,:,9),'LineWidth',1.0); hold on
yline(EP_y_sum(5,9),'--r','LineWidth',0.8); hold off
xlim([-0.1*s 1.1*s])
grid on
title('リンク(ij=5,t=9)のSAV車両数の推移')
xlabel('Day'); ylabel('SAV車両数')
lgd = legend({'SAV車両数','均衡SAV車両数'},'FontSize',14,'TextColor','black','Location','southeast');
%%}

%{
figure % p(5,10)
plot(p_5_10_rec,'LineWidth',0.8); hold on
yline(EP_p(5,10),'--r'); hold off
title('p(5,10)の推移')
xlabel('Day'); ylabel('価格')
%}

%{
figure % p(2,7))
plot(p_2_7_rec,'LineWidth',0.8); hold on
yline(EP_p(2,7),'--r'); hold off
title('p(2,7)の推移')
xlabel('Day'); ylabel('価格')
%}

%{
figure % p(7,9)
plot(p_7_9_rec,'LineWidth',0.8); hold on
yline(EP_p(7,9),'--r'); hold off
title('p(7,9)の推移')
xlabel('Day'); ylabel('価格')
%}

%{
figure % p(7,10)
plot(p_7_10_rec,'LineWidth',0.8); hold on
yline(EP_p(7,10),'--r'); hold off
title('p(7,10)の推移')
xlabel('Day'); ylabel('価格')
%}

%{
figure %トリップ(())価格
plot(p_2_7_rec+p_7_9_rec,'LineWidth',0.8); hold on
yline(EP_p(2,7)+EP_p(7,9),'--r'); hold off
title('トリップ価格(2,7)(7,9)の推移')
xlabel('Day'); ylabel('価格')
%}

%{
figure % Z_ave
plot(Z_ave_rec,'LineWidth',0.8); hold on
yline(0,'--r','LineWidth',0.8); hold off
grid on
title('市場精算過程')
xlabel('Day'); ylabel('')
%}



%% [EP-L-Dual]で発生する各最適OD経路別の最小費用
SP_U_X_variety = unique(SP_U_X_rec,'rows');
SP_U_X_num = height(SP_U_X_variety);
OD_each = zeros(SP_U_X_num,iteration);
for i=1:SP_U_X_num
    for j=1:iteration
        OD_each(i,:) = SP_U_X_variety(i,:) * (xf_co_rec.');
    end
end

acolor = zeros(SP_U_X_num,3);
for i=1:SP_U_X_num
    acolor(i,:) = [rand rand rand];
end

figure
plot(OD_each(1,:),'LineWidth',0.5,'Color',acolor(1,:)); hold on
for i=2:SP_U_X_num
    plot(OD_each(i,:),'LineWidth',0.5,'Color',acolor(i,:));
end
hold off
xlim([-0.1*s 1.1*s])
grid on
title('[EP-L-Dual]で発生する各最適OD経路別費用の推移')
xlabel('Day'); ylabel('費用')



%% 残り
SP_P
SP_U
EP_L_Dual_real
EP_L_Dual
EP