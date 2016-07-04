%

% clear
clc

% load('Indices_SVSF_14_hiddens_1000_trials.mat')
% SVSF_Indices=Indices_Matrix;
SVSF_Indices=Indices_Matrix_cad;
% load('Indices_Kalman_14_hiddens_1000_trials.mat')
% Kalman_Indices=Indices_Matrix;
Kalman_Indices=Indices_Matrix_wt;

% Index
% 1 - CPU Time
% 2 - Convergence Epochs
% 3 - Accuracy
% 4 - RMS Error

for Index=1:6;
    
    switch Index
        case 1
            index_name='CPU Time (Second per Training)';
        case 2
            index_name='Convergence Rate (Epochs)';
        case 3
            index_name='# Train';
        case 4
            index_name='# Validation';
        case 5
            index_name='# Test';
        case 6
            index_name='RMS Error';
    end
    
    x1=SVSF_Indices(:,Index);
    x2=Kalman_Indices(:,Index);

    X=[x1,x2];
    X_mean=mean(X);
    disp_content=['Index ',num2str(Index),' :',index_name];
    disp(disp_content);
    disp(' ');
    disp(X_mean);

    [p,table,stats]=anova1(X);
    c=multcompare(stats);
    disp(c);
end

close all