function net = nn_create(nn_struct)
    % Create NN
    net = network;
    
    % Number of NN layers
    size_nn = size(nn_struct);
    num_layers          = size_nn(1,2) - 1;
    net.numInputs       = 1;
    net.numLayers       = num_layers;
    
    % NN layer connections
    net.biasConnect     = ones(num_layers,1);
    net.inputConnect    = [1; zeros(num_layers-1,1)];
    net.layerConnect    = zeros(num_layers, num_layers);
    for i = 1:num_layers-1
        net.layerConnect(i+1,i) = 1;
    end
    net.outputConnect   = [zeros(1,num_layers-1) 1];
    
    % NN layer sizes
    net.inputs{1}.size  = nn_struct(1);
    for i = 1:num_layers
        net.layers{i}.size          = nn_struct(i+1);
         if i == num_layers
             net.layers{i}.transferFcn   = 'purelin';
             %net.layers{i}.transferFcn   = 'logsig';
         else
            net.layers{i}.transferFcn   = 'logsig';
        end
    end
    
    % NN initial weight and bias values
    for i = 1:num_layers
        net.layers{i}.initFcn = 'initnw';
        net = initnw(net,i);  % For some reason the init function below doesn`t do anything on my desktop
    end
    
    % Init NN
    net = init(net);
end

