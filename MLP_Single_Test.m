% ----------------------------------------------------------------------- %

% [MATALB VERSION]
% This program is written using MatLAB 2010a. Newer versions of MatLAB may
% cause unexpected problems in running it.

% [PROGRAM PURPOSE]
% This program trains the Multi-layer Perceptron (MLP) Nueral Network using
% different algorithms and with user-defined training dataset.

% [INSTRUCTION]
% 1) The user needs to provide a proper training dataset;
% 2) The user must indicate the ratio of the data for training, validation
% and test;
% 3) The user must indicate other parameters in the codes.

% ----------------------------------------------------------------------- %
clear
clc
close all

%folderFeature='C:\Users\Yifei Feng\Google Drive\3-Research\4-Matlab Program\FDD-NN\Temp_Data\';
folderFeature='C:\Users\Yifei Feng\Desktop\FDD_Data\New_720\';

feature_type='FT';
switch feature_type
    case 'CAD' % Running_Average(CAD)
        fileFeature='Feature_average_cad_cycle_1_100.mat';
    case 'FT' % Running_Average(FT(CAD))
        fileFeature='Feature_average_ft_cad_cycle_1_100.mat';
    case 'WT' % Running_Average(WT(CAD))
        fileFeature='Feature_average_wt_cad_cycle_1_100.mat';
end

load([folderFeature,fileFeature]);
Feature=Feature_Matrix; % You need to load Feature_Matrix manually.

load('Category_8_100.mat');
Category=Category_Matrix;

Ratio_Training=0.6;
Ratio_Validation=0.4;
Ratio_Test=0;

% ----------------------------------------------------------------------- %

Num_Category=size(Category,1);
Num_Sample_Per_Category=size(Category,2)/Num_Category;

Feature_Training=[];
Feature_Validation=[];
Feature_Test=[];

Category_Training=[];
Category_Validation=[];
Category_Test=[];

for i1=1:Num_Category
    
    Feature_Classified_Temp=...
        Feature(:,(i1-1)*Num_Sample_Per_Category+1:i1*Num_Sample_Per_Category);
    
    Category_Classified_Temp=...
        Category(:,(i1-1)*Num_Sample_Per_Category+1:i1*Num_Sample_Per_Category);
    
    
    [Feature_Training_Temp,Feature_Validation_Temp,Feature_Test_Temp]=...
        dividerand(Feature_Classified_Temp,Ratio_Training,Ratio_Validation,Ratio_Test);
    
    [Category_Training_Temp,Category_Validation_Temp,Category_Test_Temp]=...
        dividerand(Category_Classified_Temp,Ratio_Training,Ratio_Validation,Ratio_Test);
    
    Feature_Training=[Feature_Training,Feature_Training_Temp];
    Feature_Validation=[Feature_Validation,Feature_Validation_Temp];
    Feature_Test=[Feature_Test,Feature_Test_Temp];
    
    Category_Training=[Category_Training,Category_Training_Temp];
    Category_Validation=[Category_Validation,Category_Validation_Temp];
    Category_Test=[Category_Test,Category_Test_Temp];
end

% ----------------------------------------------------------------------- %


%Create Neural Network
num_hidden=10;
nn_struct=[size(Feature,1),num_hidden,size(Category,1)];
net=nn_create(nn_struct);

%MLP Traing Parameters
num_epoch=15;
stop_err_rms=0.017; % EMS
stop_err_grad=0.01;

train_type='svsf';

if strcmp(train_type,'lm')==1
    
    %Train neural network using LM
    lm_input=Feature;
    lm_target=Category;
    net.trainFcn='trainlm';%'traingd';%
    net.trainParam.epochs=num_epoch;
    net.trainParam.goal=stop_err_rms;%!!!!! Shoulb be EMS
    net.trainParam.min_grad=stop_err_grad;
    net.trainParam.showWindow=true;%false;%
    net.trainParam.showCommandLine=true;
    net.trainParam.show=1;
    net.divideFcn='dividerand';
    net.divideParam.trainRatio=Ratio_Training;
    net.divideParam.valRatio=Ratio_Validation;
    net.divideParam.testRatio=Ratio_Test;
    net.performFcn='mse';
    [net_train,TR]=train(net,lm_input,lm_target);
    
    err_rms_train=TR.perf;
    err_rms_val=TR.vperf;
    err_rms_test=TR.tperf;
    
    Feature_Training=lm_input(:,TR.trainInd);
    Category_Training=lm_target(:,TR.trainInd);
    Feature_Validation=lm_input(:,TR.valInd);
    Category_Validation=lm_target(:,TR.valInd);
    Feature_Test=lm_input(:,TR.testInd);
    Category_Test=lm_target(:,TR.testInd);
    
    % training_epoch,training_time missing
    
elseif strcmp(train_type,'ekf')==1
    
    %Train neural network using EKF
    
    fprintf('\nTrainingwithEKF.\n\n');
    [net_train,err_rms_train,err_rms_val,err_rms_test]=...
        nn_trainEKF_ms(net,num_epoch,stop_err_rms,stop_err_grad,...
        Feature_Training,Category_Training,Feature_Validation,Category_Validation,Feature_Test,Category_Test);

elseif strcmp(train_type,'svsf')==1
    
    %Train neural network using SVSF
    
    fprintf('\nTraining with SVSF.\n');
    conv_rate=0;%Gamma
    bound_thickness=0.1;%Psi
    fprintf('Convergence Rate:%.2g\n',conv_rate);
    fprintf('Boundary Thickness:%.2g\n\n',bound_thickness);
    [net_train,err_rms_train,err_rms_val,err_rms_test,training_epoch,training_time]=...
        nn_trainSVSF_ms(net,conv_rate,bound_thickness,num_epoch,stop_err_rms,stop_err_grad,...
        Feature_Training,Category_Training,Feature_Validation,Category_Validation,Feature_Test,Category_Test);

end

%-----------------------------------------------------------------------%

%Test Neural Network
output_train=sim(net_train,Feature_Training);
err_train=sqrt(mean((Category_Training(:)-output_train(:)).^2));

output_val=sim(net_train,Feature_Validation);
err_val=sqrt(mean((Category_Validation(:)-output_val(:)).^2));

output_test=sim(net_train,Feature_Test);
err_test=sqrt(mean((Category_Test(:)-output_test(:)).^2));

% aa=sim(net_train,Feature_Training);

[~,max_output]=max(sim(net_train,Feature_Training));
[~,max_target]=max(Category_Training);
num_failed_train=numel(find(max_output~=max_target));

[~,max_output]=max(sim(net_train,Feature_Validation));
[~,max_target]=max(Category_Validation);
num_failed_val=numel(find(max_output~=max_target));

[~,max_output]=max(sim(net_train,Feature_Test));
[~,max_target]=max(Category_Test);
num_failed_test=numel(find(max_output~=max_target));

plotconfusion(Category_Training,output_train);

plotconfusion(Category_Validation,output_val);

fprintf('\n');
fprintf('Training Results:     %.4f  (%d/%d failed)\n',err_train,num_failed_train,size(Feature_Training,2));
fprintf('Validation Results:   %.4f  (%d/%d failed)\n',err_val,num_failed_val,size(Feature_Validation,2));
fprintf('Testing Results:      %.4f  (%d/%d failed)\n',err_test,num_failed_test,size(Feature_Test,2));

% disp(['Convegence Epoch: ',num2str(training_epoch),', Time:(s)',num2str(training_time)]);

figure
plot(err_rms_train)
xlabel('Epoch');
ylabel('MSE');
hold on
plot(err_rms_val,'r')
legend('Training','Validation')
title('');
% set(gca, 'XLim', [1 20]);
set(gca, 'YLim', [0 2]);

save_folder='C:\Users\Yifei Feng\Google Drive\3-Research\4-Matlab Program\FDD-NN\Temp_Data\';
save_file_name_2='NN_network.mat';
% save([save_folder,save_file_name_2],'net_train');
