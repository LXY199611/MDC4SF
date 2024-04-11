% Conduct ablation study for test wether label fusion work

clear;
clc; 
load("data/Enb.mat");
data = getfield(data, 'norm');

q = size(target,2); % 一共几个class space
global rankN;
rankN = q;
gammaSet=[1e-2];
alphaSet=[1e-1];
lambdaSet=[0];
% para.gamma = 1e-2;
% para.lambda = 0.0001;
% para.alpha = 1e-1;
para.num_cluster = 5;
for alphaIndex = 1:length(alphaSet)
for gammaIndex = 1:length(gammaSet)
for lambdaIndex = 1:length(lambdaSet)
para.gamma = gammaSet(gammaIndex);
para.alpha = alphaSet(alphaIndex);
para.lambda = lambdaSet(lambdaIndex);
for times=1:1
    fprintf('rep:%d\n',times);
    idx = idx_folds{times};
    train_id = getfield(idx, 'train');
    test_id = getfield(idx, 'test');
    data_train = data(train_id,:);
    data_test = data(test_id,:);
    target_train = target(train_id,:); 
    target_test = target(test_id,:); 
    n = size(data_train,1);
    F_train = [];
    for i=1:q
        K{i} = max(target_train(:,i));
        label{i} = zeros(n,K{i});
        for j = 1:n
            label{i}(j,target_train(j,i))=1;
        end
        F_train = [F_train,label{i}];
    end
    lambda = 1.2;
    [Z,~] = solve_lrr(F_train,lambda);
%     Y_ = F_train*Z;
    Y_ = F_train;
    X_new = data_train;
    Kpara.type = 'RBF';%RBF kernel
    if n<2000
        Kpara.gamma  = 1/2/std(pdist(X_new))^2; %parameter of kernel function
    else
        Kpara.gamma  = 1/2/std(pdist(X_new(1:2000,:)))^2; %parameter of kernel function
    end
    
    % Kernel version
    [Theta,data_new] = simple_kernel_regression_local(F_train,X_new,Z,Kpara,para);
    % Test
    F_test = Kernel(data_test,data_train ,Kpara)*Theta;
    Y_encode = zeros(size(data_test,1),q);
    for i=1:size(data_test,1)
        start = 1;
        j=1;
        while(j<=q)
            e = start+K{j}-1;
            temp = F_test(i,start:e);
            optimal_label_index = find(temp==max(temp));
            Y_encode(i,j) = optimal_label_index;
            start = e + 1;
            j = j + 1;
        end
    end
    
    %Hamming Score(or Class Accuracy)
    Eval.HS(times) = sum(sum(Y_encode==target_test))/(size(target_test,1)*size(target_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    Eval.EM(times) = sum(sum((Y_encode==target_test),2)==size(target_test,2))/size(target_test,1);
    %Sub-ExactMatch
    Eval.SEM(times) = sum(sum((Y_encode==target_test),2)>=(size(target_test,2)-1))/size(target_test,1);
end

fprintf('Alpha:%f \t Gamma:%f \t Lambda:%f \n',para.alpha,para.gamma,para.lambda);
disp(['HammingScore=',num2str(mean(Eval.HS),'%4.3f'),', ExactMatch=',num2str(mean(Eval.EM),'%4.3f'),', SubExactMatch=',num2str(mean(Eval.SEM),'%4.3f')]);
end
end
end
