function new_W = updateW(X,Y,e,W,G_sample,index,g,param)
global   trainFeature;
global   trainLabel;
global   E;
global   para;
global num_cluster;
global Idx;
global g_sample;

g_sample = G_sample;
Idx = index;
trainFeature = X;
trainLabel = Y;
E = e;
para = param;
num_cluster = g;

[new_W,~] = fminlbfgs(@MDCbfgsProcess_W,W);

end