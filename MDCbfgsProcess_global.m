function [target,gradient] = MDCbfgsProcess_global(E)

% Update instance local correlation matrix E_l


global   trainFeature;
% global   trainLabel;
global   weights;
global   para;
% global   originX;
% global num_cluster;
% global Idx;
% global Q;

% [Idx,~] = kmeans(originX,num_cluster);  
modProb =  trainFeature * weights;  % size_sam * size_Y

gamma=para.gamma;
lambda=para.lambda;
% alpha = para.alpha;
G = E*E';

% L = sum(sum((modProb - trainLabel).^2)); 
% R2 = sum(sum((weights).^2));  % penalty
R1 = trace(modProb'*G*modProb); % global

target = lambda*R1;

% gradL = 2*trainFeature' * (modProb - trainLabel);
% gradR2 = 2*weights;
% gradR1 = trainFeature'*E*E'*trainFeature*weights + (trainFeature'*E*E'*trainFeature)'*weights;
gradR1 = 2*trainFeature*weights*(trainFeature*weights)'*E;

gradient = lambda*gradR1;

end


