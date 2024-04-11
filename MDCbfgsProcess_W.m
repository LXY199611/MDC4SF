function [target,gradient] = MDCbfgsProcess_W(weights)

% Update MDC model parameter W
global   trainFeature;
global   trainLabel;
global   E;
global   para;
global num_cluster;
global Idx;
global g_sample;

modProb =  trainFeature * weights;  % size_sam * size_Y

gamma=para.gamma;
alpha = para.alpha;
lambda = para.lambda;

L = sum(sum((modProb - trainLabel).^2)); 
R1 = trace(modProb'*g_sample*modProb); 
R2 = sum(sum((weights).^2));  % penalty
R3 = 0;
for g=1:num_cluster
    G = E{g}*E{g}';
    R3 = R3 + trace(modProb(Idx==g,:)*G*modProb(Idx==g,:)'); % local for matrix
end
target = L + lambda*R1 + gamma*R2 + alpha*R3;

gradL = 2*trainFeature' * (modProb - trainLabel);
gradR1 = trainFeature'*g_sample*trainFeature*weights + ( trainFeature'*g_sample'*trainFeature)'*weights;
gradR2 = 2*weights;
gradR3 = 0;

for g=1:num_cluster
    G = E{g}*E{g}';
    gradR3 = gradR3 + trainFeature(Idx==g,:)'*trainFeature(Idx==g,:)*weights*G' + trainFeature(Idx==g,:)'*trainFeature(Idx==g,:)*weights*G;
end
gradient = gradL + lambda*gradR1 + gamma*gradR2 + alpha*gradR3;

end


