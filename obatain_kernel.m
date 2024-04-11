fileFolder=fullfile('data/');
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name};

for index=1:length(fileNames)
    load(strcat('data/',fileNames{index}));
    name = strsplit(fileNames{index},'.');
    data = getfield(data, 'norm');
    for times=1:10
        idx = idx_folds{times};
        train_id = getfield(idx, 'train');
        test_id = getfield(idx, 'test');
        data_train = data(train_id,:);
        data_test = data(test_id,:);
        target_train = target(train_id,:); 
        target_test = target(test_id,:); 
        Kpara.type = 'RBF';%RBF kernel
        X_new = data_train;
        if n<2000
            Kpara.gamma  = 1/2/std(pdist(X_new))^2; %parameter of kernel function
        else
            Kpara.gamma  = 1/2/std(pdist(X_new(1:2000,:)))^2; %parameter of kernel function
        end
        X_train = X_new;
        K = Kernel(X_train,X_train,Kpara);%Kernel matrix
    end
end
save(strcat('kernel_data/',name{1},'.mat'),'')