% ODペアが2つを想定((出発ノード,到着ノード)=(1,4),(3,2))
% 利用者の選好情報は全員同じ（同じ希望到着時間を持つ）
% SAVプロバイダーは1社，すなわち，拠点も1つでそれぞれノード1



% 道路情報
%road = zeros(L,5); %(リンク番号，発ノード，着ノード，自由旅行時間，容量)
road = [
    1	1	2	6	65
    2	1	3	2	55
    3	2	1	6	65
    4	2	6	2	60
    5	3	1	2	55
    6	3	4	5	60
    7	3	12	5	60
    8	4	3	5	60
    9	4	5	3	50
    10	4	11	5	55
    11	5	4	3	50
    12	5	6	3	50
    13	5	9	2	50
    14	6	2	2	60
    15	6	5	3	50
    16	6	8	3	45
    17	7	8	3	40
    18	7	18	5	50
    19	8	6	3	45
    20	8	7	3	40
    21	8	9	3	45
    22	8	16	2	45
    23	9	5	2	50
    24	9	8	3	45
    25	9	10	2	45
    26	10	9	2	45
    27	10	11	5	50
    28	10	15	4	45
    29	10	16	3	40
    30	10	17	3	45
    31	11	4	5	55
    32	11	10	5	50
    33	11	12	3	60
    34	11	14	4	50
    35	12	3	5	60
    36	12	11	3	60
    37	12	13	6	65
    38	13	12	6	65
    39	13	24	2	60
    40	14	11	4	50
    41	14	15	4	50
    42	14	23	3	40
    43	15	10	4	45
    44	15	14	4	50
    45	15	19	3	40
    46	15	22	3	45
    47	16	8	2	45
    48	16	10	3	40
    49	16	17	2	45
    50	16	18	3	55
    51	17	10	3	45
    52	17	16	2	45
    53	17	19	3	45
    54	18	7	5	50
    55	18	16	3	55
    56	18	20	6	55
    57	19	15	3	40
    58	19	17	3	45
    59	19	20	4	50
    60	20	18	6	55
    61	20	19	4	50
    62	20	21	3	40
    63	20	22	4	45
    64	21	20	3	40
    65	21	22	2	50
    66	21	24	3	50
    67	22	15	3	45
    68	22	20	4	45
    69	22	21	2	50
    70	22	23	4	40
    71	23	14	3	40
    72	23	22	4	40
    73	23	24	2	40
    74	24	13	2	60
    75	24	21	3	50
    76	24	23	2	40
];



% 与件のパラメータ
N = max(road(:,2)); % ノード数
L = height(road); % リンク数
T = 500; % 時間ステップ数
t = repmat(road(:,4),1,T-1); % (L,(T-1))に対応した自由旅行時間
limit = repmat(road(:,5),1,T-1); % (L,(T-1))に対応した道路容量
C = 3; % 車両容量(人/台)
d = 100; % サービス費用(円)(今回は全てのリンクで同じ)
alfa = 30; % 時間価値(円/min)
beta = 50; % アクティブ車両数１台あたりのコスト係数(円/台) 
R = 2; % SAVプロバイダーの数
R_base = [1 3];
S = [40 30]; % 1社が保有するSAV車両数
OD = 2; % ODトリップ数
Q = [380 250]; % 各経路の利用者数
O_node = [1 18];
D_node = [10 5];
D_hope = [12 16]; % 希望到着時間
w = zeros(1,OD*T); % スケジュール費用
for i=0:OD-1
    for j=1:D_hope(1,i+1)
        w(1,T*i+j) = 1*(D_hope(1,i+1)-j)*t(1,1)*25;
    end
    for j=D_hope(1,i+1)+1:T
        w(1,T*i+j) = 2*(j-D_hope(1,i+1))*t(1,1)*25;
    end
end



% 時空間ネットワーク情報（道路情報）
% network = zeros(L,7,T);
%network = readmatrix('network.dat');



% 利用者情報
%user = zeros(Q,4); %(出発ノード=1，到着ノード=4，到着希望時刻=5，到着時刻）



% 目的変数
%x = zeros(L,(T-1)); % 利用者フロー(リンク，時刻)
%limit = repmat(road(:,5),1,T-1); % (L,(T-1))に対応した道路容量
%f = zeros(1,T*OD); % 起終点ペアの終点到着フロー(到着ノード，時刻)
%y = ones(L,(T-1)).*road(:,4); % SAV車両フロー(リンク，時刻)
%g = zeros(1,T*R);
%h = zeros(1,T*R); % アクティブSAV車両数(車両数，時刻)


% 目的変数の係数
x_co = ones(1,L*(T-1)*OD).*alfa.*t;
f_co = ones(1,T*OD).*w;
y_co = ones(1,L*(T-1)*R).*d;
g_co = zeros(1,T*R);
h_co = ones(1,T*R).*beta;
coefficient = [x_co f_co y_co g_co h_co]; % 係数のlinprog
column_long = L*(T-1)*(OD+R)+T*(OD+R*2);

% 利用者のノードのフロー保存則
node_flow_user = zeros((N-1)*T*OD,column_long);
for o=1:OD
    user_block = zeros((N-1),L*2); % 利用者フローの行列をブロック分け
    NODE = 1:N;
    NODE(:,O_node(1,o)) = [];
    if O_node(1,o)<D_node(1,o)
        D_node(1,o)=D_node(1,o)-1;
    end
    for i=1:T
        for j=1:N-1
            column = NODE(1,j);
            for k=1:L
                if road(k,2)==column
                    user_block(j,L+road(k,1)) = 1; % 流出
                end
                if road(k,3)==column
                    user_block(j,road(k,1)) = -1; % 流入
                end
            end
        end
        if i==1
            node_flow_user((N-1)*T*(o-1)+(N-1)*(i-1)+1:(N-1)*T*(o-1)+(N-1)*(i-1)+(N-1),L*(T-1)*(o-1)+L*(i-1)+1:L*(T-1)*(o-1)+L*(i-1)+L) = user_block(1:N-1,L+1:L*2);
        elseif i==T
            node_flow_user((N-1)*T*(o-1)+(N-1)*(i-1)+1:(N-1)*T*(o-1)+(N-1)*(i-1)+(N-1),L*(T-1)*(o-1)+L*(i-2)+1:L*(T-1)*(o-1)+L*(i-2)+L) = user_block(1:N-1,1:L);
        else
            node_flow_user((N-1)*T*(o-1)+(N-1)*(i-1)+1:(N-1)*T*(o-1)+(N-1)*(i-1)+(N-1),L*(T-1)*(o-1)+L*(i-2)+1:L*(T-1)*(o-1)+L*(i-2)+L*2) = user_block;
        end
        node_flow_user((N-1)*T*(o-1)+(N-1)*(i-1)+D_node(1,o),L*(T-1)*OD+T*(o-1)+i)=1; % fのところ
    end
end

% 利用者のODフロー保存則
OD_flow_user = zeros(OD,column_long);
for i=1:OD
    OD_flow_user(i,L*(T-1)*OD+T*(i-1)+1:L*(T-1)*OD+T*(i-1)+T) = ones(1,T)*(-1);
end

% SAV車両のノードのフロー保存則
node_flow_sav = zeros(N*T*R,column_long);
sav_block_a = zeros(N,L);
sav_block_b = zeros(N,L);
sav_block = [sav_block_a sav_block_b]; % 利用者フローの行列をブロック分け
for i=1:T
    for j=1:N
        for k=1:L
            if road(k,2)==j
                sav_block_b(j,road(k,1)) = 1; % 流出
            end
            if road(k,3)==j
                sav_block_a(j,road(k,1)) = -1; % 流入
            end
        end
        sav_block = [sav_block_a sav_block_b];
    end
    for j=0:R-1
        if i==1
            node_flow_sav(N*T*j+(i-1)*N+1:N*T*j+N*(i-1)+N,L*(T-1)*OD+T*OD+L*(T-1)*j+1:L*(T-1)*OD+T*OD+L*(T-1)*j+L) = sav_block_b;
        elseif i==T
            node_flow_sav(N*T*j+(i-1)*N+1:N*T*j+N*(i-1)+N,L*(T-1)*OD+T*OD+L*(T-1)*j+(i-2)*L+1:L*(T-1)*OD+T*OD+L*(T-1)*j+(i-2)*L+L) = sav_block_a;
        else
            node_flow_sav(N*T*j+(i-1)*N+1:N*T*j+N*(i-1)+N,L*(T-1)*OD+T*OD+L*(T-1)*j+(i-2)*L+1:L*(T-1)*OD+T*OD+L*(T-1)*j+(i-2)*L+L*2) = sav_block;
        end
        node_flow_sav(N*T*j+(i-1)*N+R_base(1,j+1),L*(T-1)*(OD+R)+T*OD+T*j+i)=1; % gのところ
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



% linprogを回す
A = [road_matrix; service_matrix];
A_S = sparse(A);
b = [ones(1,L*(T-1)).*reshape(limit,1,[]) zeros(1,L*(T-1))];
Aeq = [node_flow_user; OD_flow_user; node_flow_sav; OD_flow_sav; sav_end];
Aeq_S = sparse(Aeq);
beq = [zeros(1,(N-1)*T*OD) ones(1,OD).*Q.*(-1) zeros(1,N*T*R) zeros(1,T*R) zeros(1,R)];
lb_SAVnet = ones(1,T*R);
for i=1:R
    lb_SAVnet(1,T*(i-1)+1:T*i) = S(1,i);
end
lb = [zeros(1,(L*(T-1)*(OD+R)+T*OD)) lb_SAVnet.*(-1) zeros(1,T*R)];
ub_ODflow = ones(1,T*OD);
for i=1:OD
    ub_ODflow(1,T*(i-1)+1:T*i) = Q(1,i);
end
ub = [C.*repmat(reshape(limit,1,[]),1,OD) ub_ODflow repmat(reshape(limit,1,[]),1,R) lb_SAVnet lb_SAVnet];

[EPX,EP,exitflag,output,lambda] = linprog(coefficient, A_S, b, Aeq_S, beq, lb, ub);

EP
x = reshape(EPX(1:L*(T-1)*OD,1),L,(T-1)*OD);
if OD==1
    x
elseif OD==2
    x1 = x(:,1:T-1)
    x2 = x(:,(T-1)+1:(T-1)*2)
elseif OD==3
    x1 = x(:,1:T-1)
    x2 = x(:,(T-1)+1:(T-1)*2)
    x3 = x(:,(T-1)*2+1:(T-1)*3)
end
f = reshape(EPX(L*(T-1)*OD+1:L*(T-1)*OD+T*OD,1),T,OD)'
y = reshape(EPX(L*(T-1)*OD+T*OD+1:L*(T-1)*(OD+R)+T*OD,1),L,(T-1)*R);
if R==1
    y
elseif R==2
    y1 = y(:,1:T-1)
    y2 = y(:,(T-1)+1:(T-1)*2)
elseif R==3
    y1 = y(:,1:T-1)
    y2 = y(:,(T-1)+1:(T-1)*2)
    y3 = y(:,(T-1)*2+1:(T-1)*3)
end
g = reshape(EPX(L*(T-1)*(OD+R)+T*OD+1:L*(T-1)*(OD+R)+T*(OD+R),1),T,R)'
h = reshape(EPX(L*(T-1)*(OD+R)+T*(OD+R)+1:L*(T-1)*(OD+R)+T*(OD+R*2),1),T,R)'



% ラグランジュ乗数の生成
e = reshape(lambda.ineqlin(1:L*(T-1),1),L,(T-1))
p = reshape(lambda.ineqlin(L*(T-1)+1:L*(T-1)*2,1),L,(T-1))
u = reshape(lambda.eqlin(1:(N-1)*T*OD,1),N-1,T*OD);
if OD==1
    u
elseif OD==2
    u1 = u(:,1:T)
    u2 = u(:,T+1:T*2)
elseif OD==3
    u1 = u(:,1:T)
    u2 = u(:,T+1:T*2)
    u3 = u(:,T*2+1:T*3)
end
rho = lambda.eqlin((N-1)*T*OD+1:(N-1)*T*OD+OD,1)
v = reshape(lambda.eqlin((N-1)*T*OD+OD+1:(N-1)*T*OD+OD+N*T*R,1),N,T*R);
if R==1
    v
elseif R==2
    v1 = v(:,1:T)
    v2 = v(:,T+1:T*2)
elseif R==3
    v1 = v(:,1:T)
    v2 = v(:,T+1:T*2)
    v3 = v(:,T*2+1:T*3)
end
phi = reshape(lambda.eqlin((N-1)*T*OD+OD+N*T*R+1:(N-1)*T*OD+OD+N*T*R+T*R,1),T,R)'
v_dif = (-1)*phi
pi = reshape(lambda.upper((T-1)*L*(OD+R)+T*(OD+R)+1:(T-1)*L*(OD+R)+T*(OD+R*2),1),T,R)'