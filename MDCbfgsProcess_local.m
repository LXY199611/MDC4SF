function [target,gradient] = MDCbfgsProcess_local(E)

% Update instance local correlation matrix E_l
global   trainFeature;
global   weights;
global   para;
global Idx
global current_num;


modProb =  trainFeature * weights;  % size_sam * size_Y
alpha = para.alpha;
G = E*E';
R3 = trace(modProb(Idx==current_num,:)*G*modProb(Idx==current_num,:)'); % local


target = alpha*R3;

gradR3 = 2*(trainFeature(Idx==current_num,:)*weights)'*(trainFeature(Idx==current_num,:)*weights)*E;

gradient = alpha*gradR3;

end


