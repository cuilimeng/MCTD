clear all
close all
clc

load('nus_wide_data.mat')

%% 选出只属于一个类别的样本点
L_train = L_tr(sum(L_tr,2)==1,:);
I_tr = I_tr(sum(L_tr,2)==1,:);
T_tr = T_tr(sum(L_tr,2)==1,:);
L_test = L_te(sum(L_te,2)==1,:);
I_te = I_te(sum(L_te,2)==1,:);
T_te = T_te(sum(L_te,2)==1,:);

tr_num = size(L_train,1);
te_num = size(L_test,1);
L_tr = zeros(tr_num,1);
L_te = zeros(te_num,1);
for i = 1:tr_num
    L_tr(i) = find(L_train(i,:) == 1);
end
for i = 1:te_num
    L_te(i) = find(L_test(i,:) == 1);
end

%% 确定训练集和测试集都有类别的样本点
cat_ind = intersect(L_tr,L_te);
tr_ind = ismember(L_tr,cat_ind);
te_ind = ismember(L_te,cat_ind);
I_tr = I_tr(tr_ind,:);
I_te = I_te(te_ind,:);
T_tr = T_tr(tr_ind,:);
T_te = T_te(te_ind,:);
L_tr = L_tr(tr_ind,:);
L_te = L_te(te_ind,:);
L_train = L_train(tr_ind,:);
L_test = L_test(te_ind,:);

%% feature归一化至[0,1]
I_tr = mapminmax(I_tr,0,1);
I_te = mapminmax(I_te,0,1);
T_tr = mapminmax(T_tr,0,1);
T_te = mapminmax(T_te,0,1);

%% 重新定义类别label
cat_num = size(cat_ind,1);
for i = 1:cat_num
    L_tr(L_tr == cat_ind(i))=i;
    L_te(L_te == cat_ind(i))=i;
end

L_train = zeros(size(L_tr,1),cat_num);
L_test = zeros(size(L_te,1),cat_num);
for i = 1:size(L_tr,1)
    L_train(i,L_tr(i)) = 1;
end
for i = 1:size(L_te,1)
    L_test(i,L_te(i)) = 1;
end

%% 统计每一类样本点的数目。ca的第一列训练集，第二列测试集
ca = zeros(cat_num,2);
for i = 1:cat_num
    ca(i,1) = sum(L_tr == i);
    ca(i,2) = sum(L_te == i);
end
[sort_tr_ca,sort_tr_ind] = sort(ca(:,1),'descend');
[sort_te_ca,sort_te_ind] = sort(ca(:,2),'descend');

%% 保存数据
fname = 'new_nus_wide_data';
save(fname,'I_tr','I_te','T_tr','T_te','L_tr','L_te','L_train','L_test','-v7.3');

