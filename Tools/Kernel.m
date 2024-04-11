function [ K_X1_X2 ] = Kernel( X1, X2, para )
%Kernel computes kernelized inner product between instance matrix X1 and X2
%
%	Syntax
%
%       [ K_X1_X2 ] = Kernel( X1, X2, para )
%
%	Description
%
%	Kernel takes
%       X1      - An n1xd array, the ith instance of X1 is stored in X1(i,:)
%       X2      - An n2xd array, the ith instance of X2 is stored in X2(i,:)
%       para    - A struct variable that stores kernel's parameters,
%                 1)  if  kernel's type is radial basis function: exp(-gamma*|x(i)-x(j)|^2)
%                       para.type = 'RBF' while para.gamma gives the value of gamma
%                 2)  if  kernel's type is polynomial: (gamma*u'*v + coef0)^degree
%                       para.type = 'Poly' while para.gamma, para.coef0, para.degree give the value of gamma,coef0, degree respectively
%                 3)  if  kernel's type is linear: u'*v
%                       para.type = 'Linear' (default)
%	and returns,
%       K_X1_X2 - kernelized inner product between instance matrix X1 and X2

    if nargin<3
        para.type = 'Linear';
    end
    switch para.type
        case 'RBF'
            gamma = para.gamma;
            X1_inner_prod = sum(X1.^2,2);
            X2_inner_prod = sum(X2.^2,2);
            X1_X2_inner_prod = X1*X2';
            X1_inner_prod_concur = repmat(X1_inner_prod,1,size(X2_inner_prod,1));
            X2_inner_prod_concur = repmat(X2_inner_prod,1,size(X1_inner_prod,1));
            X1_minus_X2_norm_square = X1_inner_prod_concur + X2_inner_prod_concur' - 2*X1_X2_inner_prod;
            K_X1_X2 = exp(-gamma*X1_minus_X2_norm_square);
        case 'Poly'
            gamma = para.gamma;
            coef0 = para.coef0;
            degree = para.degree;
            K_X1_X2 = (gamma*X1*X2' + coef0).^degree;
        case 'Linear'
            K_X1_X2 = X1*X2';
        otherwise
            error('Kernel types not supported, please type "help Kernel" for more information');
    end
end