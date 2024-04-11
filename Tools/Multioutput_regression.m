function [ Theta,M ] = Multioutput_regression( y_train,X_train,Kpara,Rpara )
%Multioutput_regression learns the multi-output regression model in Subsection 3.3
%
%	Syntax
%
%       [ Theta,M ] = Multioutput_regression( y_train,X_train,Kpara,Rpara )
%
%	Description
%
%	Multioutput_regression takes
%       X_train - An mxd array, the ith training instance is stored in X_train(i,:)
%       y_train - An mxell array, the real-valued label vector of ith training instance is stored in y_train(i,:)
%       Kpara	- A struct variable that stores kernel parameters,
%                 1)  if  kernel's type is radial basis function: exp(-gamma*|x(i)-x(j)|^2)
%                       para.type = 'RBF' while para.gamma gives the value of gamma
%                 2)  if  kernel's type is polynomial: (gamma*u'*v + coef0)^degree
%                       para.type = 'Poly' while para.gamma, para.coef0, para.degree give the value of gamma,coef0, degree respectively
%                 3)  if  kernel's type is linear: u'*v
%                       para.type = 'Linear'
%       Rpara   - A struct variable that stores regression model's hyperparameters except its kernel parameters
%               Rpara.gamma: trade-off parameter in Eq.(9) (default 10)
%               Rpara.mu: trade-off parameter in Eq.(10) (default 1)
%               Rpara.numK: number of nearest neighbors considered in Eq.(10) (default 6)
%               Rpara.file_id corresponds to the file identifier (default 1, i.e., output to screen)
%               Rpara.head_str corresponds to the head string (default '   ');
%               Rpara.verbose: 1-outputs, 0-no ouputs  (default 1)
%	and returns,
%       Theta	- An mxell array, kernelized model parameter obtained by solving Eq.(11)
%       M       - An ellxell array, distance matrix obtained by solving Eq.(15)

    if nargin<4
        gamma = 10;
        mu = 1;
        numK = 6;
        file_id = 1;
        head_str = '      ';
        verbose = 1;
    else
        gamma = Rpara.gamma;
        mu = Rpara.mu;
        numK = Rpara.numK;
        file_id = Rpara.file_id;
        head_str = Rpara.head_str;
        verbose = Rpara.verbose;
    end
    
    %initialize parameters
    m = size(X_train,1);%#training examples
    K = Kernel(X_train,X_train,Kpara);%Kernel matrix
    ell = size(y_train,2);%number of output variables
    [~, dist_idx_train, numK_out] = KNN_matrix_label(y_train,y_train,numK);
    if numK_out<numK
        numK = numK_out;
    end
    
    %main loop
    M = eye(ell);%initialize distance metric matrix
    for iter = 1:3%the alternating procedure usually converges in 3 iterations
        %(1) Calculating W (Theta) when M is fixed
        if verbose
            temp_str = [head_str,'[Iter ',num2str(iter),'](1) Calculating W when M is fixed (',disp_time(clock,0),')...\n'];
            fprintf(file_id,temp_str);
        end
        %%solve sylvester equation AX + XB = C
        S_A = pinv(K'*K)*K*gamma*m;
        S_B = M'+M;
        S_C = S_A*y_train*(M'+M)/(gamma*m);
        Theta = sylvester(S_A,S_B,S_C);clear S_A S_B S_C;
        
        %(2) Calculating M when W (Theta) is fixed
        if verbose
            temp_str = [head_str,'[Iter ',num2str(iter),'](2) Calculating M when W is fixed (',disp_time(clock,0),')...\n'];
            fprintf(file_id,temp_str);
        end
        %(2.1)compute U according to Eq.(13)
        tmp_err = K*Theta-y_train;%size: m*ell
        U = tmp_err'*tmp_err/m;%size: ell*ell
        %(2.2)compute V according to Eq.(14)
        V = zeros(ell,ell);
        for k = 1:numK
            tmp_y = y_train(dist_idx_train(:,k),:);
            tmp_err = K*Theta-tmp_y;
            V = V + tmp_err'*tmp_err;
        end
        V = V/numK/m;%size: ell*ell
        %(2.3)compute M according to Eq.(15)
        M = Cholesky_Schur_GeometricMean(pinv(U),V,0.5,mu);
    end
end

