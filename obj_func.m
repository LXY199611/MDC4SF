function [loss] = obj_func(X,Y,E,G_sample,W,Idx,num_cluster,para)


modProb =  X * W;  % size_sam * size_Y

gamma = para.gamma;
alpha = para.alpha;
lambda = para.lambda;

L = sum(sum((modProb - Y).^2)); 
R1 = trace(modProb'*G_sample*modProb); % sample correlation
R2 = sum(sum((W).^2));  % penalty
R3 = 0;
for g=1:num_cluster
    G = E{g}*E{g}';
    R3 = R3 + trace(modProb(Idx==g,:)*G*modProb(Idx==g,:)'); % local
end
loss = L + lambda*R1 + gamma*R2 + alpha*R3;

end