% ----------------------------------------------------------------------- %

% [MATALB VERSION]
% This program is written using MatLAB 2010a. 

% [PROGRAM PURPOSE]
% This program creates the Category/Target dataset/matrix that is used to 
% train the Multi-layer Perceptron (MLP) Nueral Network.

% [INSTRUCTION]
% 1) The user needs to provide a proper training dataset;
% ----------------------------------------------------------------------- %

Num_Category=8;
Num_Sample_Per_Category=100;

save_file_name=['Category_',num2str(Num_Category),'_',num2str(Num_Sample_Per_Category),'.mat'];

Category_Matrix=zeros(Num_Category,(Num_Category*Num_Sample_Per_Category));

for i1=1:(Num_Category*Num_Sample_Per_Category)
   Index_Category=floor((i1-1)/Num_Sample_Per_Category)+1;
   Category_Matrix(Index_Category,i1)=1;
end

fprintf('Category Matrix Created!\n');

save(save_file_name,'Category_Matrix');