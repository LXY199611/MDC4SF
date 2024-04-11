function [ W, K, losslist ] = simple_kernel_regression_local(y_train,X_train,Kpara,p)
    num_cluster = p.num_cluster;
    para.gamma = p.gamma;
    para.alpha = p.alpha;
    para.lambda = p.lambda;
%     tic;
   %% sample correlation matrix build
    Idx = knnsearch(X_train,X_train,'K',10);
    GraphConnect = zeros(size(X_train,1),size(X_train,1));
    for i = 1:size(X_train,1)
        GraphConnect(i,Idx(i,:)) = 1;
    end
    GraphConnect = GraphConnect + GraphConnect';
    GraphConnect(GraphConnect > 0) = 1;
%     sigma = Kpara.gamma;
    sigma = p.sigma;
    A =  exp(-(L2_distance(X_train', X_train').^2) / (2 * sigma ^ 2));
    A = A .* GraphConnect;
    A = A - diag(diag(A));
    A_hat = diag(sum(A,2));
    G_sample = A_hat - A;
%     fprintf('sample correlation matrix build time use: %f\n',toc)
   %% initial local correlation matrix
    for i=1:num_cluster
        E{i} = eye(size(y_train,2));
    end
   %% kernel trick and clustering
%     tic;
    K = Kernel(X_train,X_train,Kpara);%Kernel matrix
%     fprintf('kernel trick time use: %f\n',toc)
    W = eye(size(K,2),size(y_train,2));
    max_iter = 100;
    iter = 0;
%     tic;
    [Idx,~] = kmeans(X_train,num_cluster); 
%     fprintf('clustering time use: %f\n',toc)
    %% alternatingly update parameters
    tic;
    loss1 = obj_func(K,y_train,E,G_sample,W,Idx,num_cluster,para);
    losslist(1)=loss1;
    while(iter < max_iter)
        iter = iter + 1;
        %
        tic;
        W = updateW(K,y_train,E,W,G_sample,Idx,num_cluster,para); % update MDC model
%         fprintf('update W time use: %f\n',toc);
        % q,X,Y,origin,e,W,index,g,param
%         tic;
        for q=1:num_cluster
            E{q} = updateE(K,E{q},W,Idx,q,para); % update local label correlation matrix G_label  = E*E'
            E{q} = NormalizeFea(E{q}); % projection
        end
%         fprintf('update E time use: %f\n',toc);
        loss2 = obj_func(K,y_train,E,G_sample,W,Idx,num_cluster,para);
        losslist(iter) = loss2;
%         fprintf('iter:%d \t loss: %.4f\n', iter, loss2);
        if (abs(loss1-loss2) < 1e-6 && loss2<loss1)
            break;
        else
            loss1 = loss2;
        end
    end
%     fprintf('ADMM time use: %f\n',toc);
end