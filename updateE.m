function new_E = updateE(X,e,W,index,q,param)
global   trainFeature;
global   weights;
global   para;
global Idx
global current_num;

current_num = q;
Idx = index;
trainFeature = X;
weights = W;
E = e;
para = param;

[new_E,~] = fminlbfgs(@MDCbfgsProcess_local,E);

end