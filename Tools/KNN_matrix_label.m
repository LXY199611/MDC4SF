function [ dist_val, dist_idx, Num ] = KNN_matrix_label( y_train,y_test,Num_in,max_mat_elements )
%KNN_matrix_label identifies a total of Num unique neighbors in y_train for each label vector (i.e., row) in y_test 

    %default parameters setting
    if nargin < 4
        max_mat_elements=1000*1000;
    end
    if nargin < 3
        Num_in = 6;
    end
    if nargin < 2
        error('Not enough input parameters!');
    end
    
    %initialize parameters
    [C_unique,ia_unique,~] = unique(y_train,'rows');%identify unique label combinations in y_train
    num_training = size(ia_unique,1);%number of label vector in training set, only consider unique label combinations
    num_testing = size(y_test,1);%number of label vector in testing set
    
    block_size = floor(max_mat_elements/num_training);%block_size*num_training <=max_mat_elements
    num_blocks = ceil(num_testing/block_size);
    
    %initialize outputs
    if num_training-1<Num_in
        Num = num_training-1;%if num_training is less than the input parameter Num_in
    else
        Num = Num_in;
    end
    dist_val = zeros(num_testing,Num);
    dist_idx = zeros(num_testing,Num);
    
    %main loop
    for iter=1:num_blocks
        %determine starting index low & ending index high of this loop
        low=block_size*(iter-1)+1;
        if(iter==num_blocks)
            high=num_testing;
        else
            high=block_size*iter;
        end
        %determine sub-distance matrix
        tmp_data=y_test(low:high,:);
        tmp_size=size(tmp_data,1);
        mat1=repmat(sum(C_unique.^2,2),1,tmp_size);
        mat2=repmat(sum(tmp_data.^2,2),1,num_training)';
        tmp_dist_matrix=mat1+mat2-2*C_unique*tmp_data';
        tmp_dist_matrix=sqrt(tmp_dist_matrix);
        tmp_dist_matrix=tmp_dist_matrix';
        for i=low:high
            %sort the distances between instane #i and intances in training set in ascending order
            [temp,index]=sort(tmp_dist_matrix(i-low+1,:));
            num_zeros = sum(temp==0);%identify neighbors which are different from itself 
            dist_val(i,:) = temp(num_zeros+1:num_zeros+Num);
            dist_idx(i,:) = ia_unique(index(num_zeros+1:num_zeros+Num));%ia_unique stores the index set w.r.t. C_unique
        end
    end
end

