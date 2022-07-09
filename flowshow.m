% gurobiじゃない方

num=0;
flowdata=zeros(300,10);
for i=1:T-1
    for j=1:L
        if EP_y_sum(j,i)>0
            num=num+1;
            %x_flowdata(num,i,road(j,2),road(j,3),road(j,4),EP_x_sum(j,i));
            flowdata(num,1)=num; %番号
            flowdata(num,2)=i; %時間帯
            flowdata(num,3)=road(j,2); %発ノード
            flowdata(num,4)=road(j,3); %着ノード
            flowdata(num,5)=road(j,4); %自由旅行時間
            flowdata(num,6)=road(j,5); %容量
            flowdata(num,7)=EP_y_sum(j,i); %台数
            flowdata(num,8)=EP_x_sum(j,i); %人数
            flowdata(num,9)=EP_e(j,i); %通行権価格
            flowdata(num,10)=EP_p(j,i); %MS価格
        end
    end
end

a=(EP_v_dif-beta)';
EP_h_column = EP_h';
EP_pi_column = EP_pi';