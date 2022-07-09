%　時刻Tに出発リンク分の変数を考えてしまっているから，余分なものがある．

% ODペアが１つを想定
% 利用者の選好情報は全員同じ（同じ希望到着時間を持つ）
% SAVプロバイダーは１社，すなわち，拠点も１つ

% 与件のパラメータ
N = 4; % ノード数
L = 9; % リンク数
T = 12; % 時間ステップ数
Q = 300; % 総利用者数
S = 120; % SAV車両数
t = 5; % 自由旅行時間(min)(今回は全てのリンクで同じ)
C = 3; % 車両容量(人/台)
d = 100; % サービス費用(円)(今回は全てのリンクで同じ)
alfa = 30; % 時間価値(円/min)
beta = 50; % アクティブ車両数１台あたりのコスト係数(円/台) 
R = 1; % SAVプロバイダーの数
OD = 1; % ODトリップ数
D_hope = 4; % 希望到着時間
w = zeros(1,T); % スケジュール費用
for i=1:D_hope
    w(1,i) = 1*(D_hope-i)*t*25;
end
for i=D_hope+1:T
    w(1,i) = 2*(i-D_hope)*t*25;
end


% 時空間ネットワーク情報（道路情報）
% network = zeros(L,7,T);
%network = readmatrix('network.dat');


% 利用者情報
user = zeros(Q,4); %(出発ノード=1，到着ノード=4，到着希望時刻=5，到着時刻）

% 道路情報
%road = zeros(L,4); %(リンク番号，発ノード，着ノード，容量)
road = [1 1 2 80
    2 1 3 30
    3 2 1 80
    4 2 3 40
    5 2 4 40
    6 3 1 30
    7 3 4 70
    8 4 2 40
    9 4 3 70];

% 目的変数
x = zeros(L,T); % 利用者フロー(リンク，時刻)
limit = repmat(road(:,4),1,T); % (L,T)に対応した道路容量
f = zeros(1,T); % 起終点ペアの終点到着フロー(到着ノード，時刻)
y = ones(L,T).*road(:,4); % SAV車両フロー(リンク，時刻)  
h = zeros(1,T); % アクティブSAV車両数(車両数，時刻)


% 目的変数の係数
x_co = ones(1,L*T).*alfa.*t;
f_co = ones(1,T).*w;
y_co = ones(1,L*T).*d;
g_co = zeros(1,R*T);
h_co = ones(1,R*T).*beta;
coefficient = [x_co f_co y_co g_co h_co]; % 係数のlinprog
% size(coefficient)
column_long = L*T*2+OD*T+R*T*2;

% 利用者のノードのフロー保存則
node_flow_user = zeros(T*(N-1),column_long);
user_block = zeros((N-1),L*2); % 利用者フローの行列をブロック分け
for i=1:T
    for j=2:N
        for k=1:L
            if road(k,2)==j
                user_block(j-1,L+road(k,1)) = 1; % 流出
            end
            if road(k,3)==j
                user_block(j-1,road(k,1)) = -1; % 流入
            end
        end
    end
    if i==1
        node_flow_user((i-1)*(N-1)+1:(i-1)*(N-1)+(N-1),(i-1)*L+1:(i-1)*L+L) = user_block(1:N-1,L+1:L*2);
    elseif i==T
        node_flow_user((i-1)*(N-1)+1:(i-1)*(N-1)+(N-1),(i-2)*L+1:(i-2)*L+L) = user_block(1:N-1,1:L);
    else
        node_flow_user((i-1)*(N-1)+1:(i-1)*(N-1)+(N-1),(i-2)*L+1:(i-2)*L+L*2) = user_block;
    end
    node_flow_user((i-1)*(N-1)+(N-1),T*L+i)=1; % fのところ
end
node_flow_user;

% 利用者のODフロー保存則
OD_flow_user = zeros(1,column_long);
OD_flow_user(1,T*L+1:T*L+T) = ones(1,T)*(-1);

% SAV車両のノードのフロー保存則
node_flow_car = zeros(T*N,column_long);
car_block = zeros(N,L*2); % 利用者フローの行列をブロック分け
car_block_a = zeros(N,L);
car_block_b = zeros(N,L);
car_block = [car_block_a car_block_b];
for i=1:T
    for j=1:N
        for k=1:L
            if road(k,2)==j
                car_block_b(j,road(k,1)) = 1; % 流出
            end
            if road(k,3)==j
                car_block_a(j,road(k,1)) = -1; % 流入
            end
        end
        car_block = [car_block_a car_block_b];
    end
    if i==1
        node_flow_car((i-1)*N+1:(i-1)*N+N,(L+OD)*T+1:(L+OD)*T+L) = car_block_b;
    elseif i==T
        node_flow_car((i-1)*N+1:(i-1)*N+N,(L+1)*T+(i-2)*L+1:(L+1)*T+(i-2)*L+L) = car_block_a;
    else
        node_flow_car((i-1)*N+1:(i-1)*N+N,(L+1)*T+(i-2)*L+1:(L+1)*T+(i-2)*L+L*2) = car_block;
    end
    node_flow_car((i-1)*N+1,(L*2+1)*T+i)=1; % gのところ
end
node_flow_car = node_flow_car*(-1);
node_flow_car(:,(L+OD)*T+1:(L*2+OD)*T+T);
car_block;

% SAV車両のODフロー保存則(hとgであらわされるもの)
OD_flow_car = zeros(T,column_long);
for i=1:T
    OD_flow_car(i,(L*2+OD)*T+1:(L*2+OD)*T+i) = ones(1,i); % gのところ
    OD_flow_car(i,(L*2+2)*T+i) = 1; % hのところ
end

% 時刻TでSAV車両は0
car_end = zeros(1,column_long);
car_end(1,column_long) = 1;

% サービス供給制約用の行列
service_matrix = zeros(L*T,column_long);
size(service_matrix)
for i=1:T
    for j=1:L
        service_matrix((i-1)*L+j,(i-1)*L+j) = 1;
        service_matrix((i-1)*L+j,(L+OD)*T+(i-1)*L+j) = C*(-1);
    end
end

% linprogを回す
A = service_matrix;
b = zeros(1,L*T);
Aeq = [node_flow_user; OD_flow_user; node_flow_car; OD_flow_car; car_end];
beq = [zeros(1,(N-1)*T) ones(1,OD)*Q*(-1) zeros(1,N*T) zeros(1,T) 0];
lb = [zeros(1,(L*2+OD)*T) ones(1,R*T).*(-1).*S zeros(1,R*T)];
ub = [C*reshape(limit,1,[]) ones(1,T*OD).*Q reshape(limit,1,[]) ones(1,R*T*2).*S];

[EPX,EP,exitflag,output,lambda] = linprog(coefficient, A, b, Aeq, beq, lb, ub);
x = reshape(EPX(1:L*T,1),L,T)
f = reshape(EPX(L*T+1:(L+OD)*T,1),1,T)
y = reshape(EPX((L+OD)*T+1:(L*2+OD)*T,1),L,T)
g = reshape(EPX((L*2+OD)*T+1:(L*2+OD+R)*T,1),1,T)
h = reshape(EPX((L*2+OD+R)*T+1:(L*2+OD+R*2)*T,1),1,T)
EP


% ラグランジュ乗数の生成
p = reshape(lambda.ineqlin,L,T)
u = reshape(lambda.eqlin(1:T*(N-1),1),N-1,T)
rho = lambda.eqlin(T*(N-1)+OD,1)
v = reshape(lambda.eqlin(T*(N-1)+OD+1:T*(N*2-1)+OD,1),N,T)
phi = reshape(lambda.eqlin(T*(N*2-1)+OD+1:T*(N*2-1+R)+OD,1),1,T)
v_dif = (-1)*phi
e = reshape(lambda.upper(T*(L+OD)+1:T*(L*2+OD),1),L,T)
pi = reshape(lambda.upper(T*(L*2+OD+R)+1:T*(L*2+OD+R*2),1),R,T)