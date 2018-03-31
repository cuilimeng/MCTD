function Data = fea_intac(data_v1,data_v2)
% generate the full-order interaction
data_num = size(data_v1,1);
num_view = 2;
data_view = cell(num_view,1);

% view 1: category information
data_view{1} = data_v1;
% view 2: image feature
data_view{2} = data_v2;

view_ind = [size(data_view{1},2),size(data_view{2},2)];

% feature interaction
Data = zeros(data_num,view_ind(1)*view_ind(2));
for i = 1:data_num
    Data(i,:) = reshape(data_view{1}(i,:)' * data_view{2}(i,:),1,view_ind(1)*view_ind(2));
end
end