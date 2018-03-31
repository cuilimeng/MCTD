%% Structured Tensor Decomposition Hashing for Wiki dataset
% Notation:
% alpha: trade off between different modalities
% beta: trade off between matrix factorization and linean projection
% mu: trade off parameter of diagonal constraints term 
% gamma: the tradeoff parameters of the mixed graph regularization term
% lambda: the tradeoff parameter of the regularization term
% k_nn: the number of nearest neighbors
%
clc
clear all;

load 'new_wiki_data.mat';
fprintf('Wiki dataset loaded...\n');

%% full-order feature interactions
% % image feature interactions
% I_tr = fea_intac(I_tr_v1,I_tr_v2);
% I_te = fea_intac(I_te_v1,I_te_v2);
% clear I_tr_v1 I_tr_v2 I_te_v1 I_te_v2
% text feature interactions
% T_tr_v1 = mapminmax(T_tr_v1,0,1);
% T_te_v1 =  mapminmax(T_te_v1,0,1);
% T_tr = T_tr_v1;
% T_te = T_te_v1;
T_tr = fea_intac(T_tr_v1,T_tr_v2);
T_te = fea_intac(T_te_v1,T_te_v2);
clear T_tr_v1 T_tr_v2 T_te_v1 T_te_v2

%% parameter settings
run = 10;
map = zeros(run,2);
para.alpha = 0.5;
para.beta = 100;
para.gamma = 1;
para.mu = 1;
para.lambda = 0.05;
para.bits = 64;
para.class_num = max(L_tr); % number of category
para.tr_n_c = zeros(1,para.class_num+1); % instance number of each category
para.te_n_c = zeros(1,para.class_num+1);
for i = 1:para.class_num
    para.tr_n_c(i+1) = sum(L_tr == i);
    para.te_n_c(i+1) = sum(L_te == i);
end
para.sub_dim = 2; % dimension of subspace
k_nn = 5;

%% arrange the data according to their labels
[~,te_indx] = sort(L_te);
[~,tr_indx] = sort(L_tr);
I_te = I_te(te_indx,:);
I_tr = I_tr(tr_indx,:);
T_te = T_te(te_indx,:);
T_tr = T_tr(tr_indx,:);
L_te = L_te(te_indx,:);
L_tr = L_tr(tr_indx,:);
L_test = L_test(te_indx,:);
L_train = L_train(tr_indx,:);

%% centralization
fprintf('centralizing data...\n');
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

%% mixed graph regularization term
W_img = adjacency(I_tr,'nn',k_nn);   % model the intra-modal similarity in image modality
W_txt = adjacency(T_tr,'nn',k_nn);   % model the intra-modal similarity in text modality
W_inter = L_train * L_train';        % model the label cosistency between the image and text modality

[row, col] = size(I_tr);
D_img = zeros(row,row);
D_txt = zeros(row,row);
D_inter = zeros(row,row);

W_img(W_img~=0) = 1;
W_txt(W_txt~=0) = 1;

for i = 1:row
    D_img(i,i) = sum(W_img(i,:));
    D_txt(i,i) = sum(W_txt(i,:));
    D_inter(i,i) = sum(W_inter(i,:));
end

L_i = D_img-W_img;
L_t = D_txt-W_txt;
L_inter = D_inter-W_inter;
L = L_i + L_t + L_inter;

for i = 1 : run
tic
fprintf('\n');
fprintf('run %d starts...\n', i);
I_temp = I_tr';
T_temp = T_tr';
I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));
Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';

%% solve the objective function
fprintf('start solving STDH...\n');
[U1, U2, P1, P2, Z, I, V] = solveSTDH(I_temp, T_temp, L, para);

%% extend to the whole database
n_c = cumsum(para.tr_n_c);
A = 2 * (para.alpha * U1' * U1 + (1- para.alpha) * U2' * U2 + 2 * para.beta * eye(para.bits)...
    + para.lambda * eye(para.bits));
B = para.gamma * (L + L');
C = -2 * (para.alpha * U1' * I_temp + (1 - para.alpha) * U2' * T_temp + para.beta * (P1 * I_temp ...
    + P2 * T_temp));
for k = 1:para.class_num
    A1 = A + para.mu * Z' * diag(I(:,n_c(k)+1)) * Z;
    B1 = B(n_c(k)+1:n_c(k+1),n_c(k)+1:n_c(k+1));
    C1 = C(:,n_c(k)+1:n_c(k+1));
    tmp_V = lyap(A1,B1,C1);
    S_total(:,n_c(k)+1:n_c(k+1)) = tmp_V;
end

%% calculate hash codes
Yi_tr = sign((bsxfun(@minus, S_total , mean(V,2)))');
Yi_te = sign((bsxfun(@minus,P1 * Im_te' , mean(V,2)))');
Yt_tr = sign((bsxfun(@minus, S_total , mean(V,2)))');
Yt_te = sign((bsxfun(@minus,P2 * Te_te' , mean(V,2)))');
toc
tic
%% evaluate
fprintf('start evaluating...\n');
simti = Yi_tr * Yt_te';
simit = Yt_tr * Yi_te';
map(i, 1) = mAP(simti,L_tr,L_te);
map(i, 2) = mAP(simit,L_tr,L_te);
fprintf('mAP at run %d runs for TextQueryOnImageDB: %.4f\n', i, map(i, 1));
fprintf('mAP at run %d runs for ImageQueryOnTextDB: %.4f\n', i, map(i, 2));
toc
end
mean(map);
fprintf('average map over %d runs for TextQueryOnImageDB: %.4f\n', run, mean(map( : , 1)));
fprintf('average map over %d runs for ImageQueryOnTextDB: %.4f\n', run, mean(map( : , 2)));