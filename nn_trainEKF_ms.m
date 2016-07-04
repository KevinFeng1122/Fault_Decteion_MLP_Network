function [net, err_rms_train, err_rms_val, err_rms_test] = nn_trainEKF_ms(net, num_epoch, stop_err_rms, stop_err_grad, input_train, target_train, input_val, target_val, input_test, target_test)
    % initial aposteriori values
    w_post              = getx(net); % getwb new version
    p_post              = 0.001*ones(numel(w_post),numel(w_post));
    noise_cov           = 0.1;
    learn_rate          = 1;
    
    % keep track of some things
    start_time          = cputime;
    val_fail            = 0;
    val_stop            = 2;
    
    % Calculate and display initial rms error
    output_train            = sim(net,input_train);
    output_val              = sim(net,input_val);
    output_test             = sim(net,input_test);
    err_rms_train   = [];   err_rms_train(end+1)    = sqrt(mean((target_train(:) - output_train(:)).^2));
    err_rms_val     = [];   err_rms_val(end+1)      = sqrt(mean((target_val(:)   - output_val(:)).^2));
    err_rms_test    = [];   err_rms_test(end+1)     = sqrt(mean((target_test(:)  - output_test(:)).^2));
    err_rms_val_best = err_rms_val;
    fprintf('Epoch\tTraining\t\tValidation\t\tVal Test\tTesting\n');
    fprintf(' 0\t%.4f            \t%.4f            \t 0/%d\t\t%.4f\n',err_rms_train,err_rms_val,val_stop,err_rms_test);
   
    for j = (1:num_epoch)
        % Randomize order of input and target vectors
        order_rand      = randperm(size(input_train,2));
        input_rand      = input_train(:,order_rand);
        target_rand     = target_train(:,order_rand);
        
        % Calculate Jacobian
        jac             = nn_jac(net,input_rand);

        % A-priori
        z_pre           = sim(net,input_rand);
        w_pre           = w_post;
        p_pre           = p_post + 0.01*eye(size(jac,2));

        % SVSF gain calculation
        k_ekf           = p_pre*jac'*inv(jac*p_pre*jac' + noise_cov*eye(size(jac,1)));
            
        % A-posteriori
        err             = target_rand - z_pre;
        err_col         = reshape(err',numel(err),1);
        w_post          = w_pre + learn_rate*k_ekf*err_col;
        p_post          = p_pre - k_ekf*jac*p_pre;
        net             = setx(net,w_post); %setwb for new version
            
        % Calculate RMS Errors and Gradients
        output_train            = sim(net,input_train);
        output_val              = sim(net,input_val);
        output_test             = sim(net,input_test);
    	err_rms_train(end+1)    = sqrt(mean((target_train(:) - output_train(:)).^2));
    	err_rms_val(end+1)      = sqrt(mean((target_val(:)   - output_val(:)).^2));
        err_rms_test(end+1)     = sqrt(mean((target_test(:)  - output_test(:)).^2));
        err_grad_train          = err_rms_train(end) - err_rms_train(end-1);
        err_grad_val            = err_rms_val(end)   - err_rms_val(end-1);
        err_grad_test           = err_rms_test(end)  - err_rms_test(end-1);
                
        % Validation Check
        if err_rms_val(end) > err_rms_val_best && err_grad_train < 0
            val_fail = val_fail+1;
        else
            val_fail = 0;
            err_rms_val_best = err_rms_val(end);
            net_best = net;
        end
        
        % Display RMS Errors and Gradients
        fprintf(' %d\t%.4f (%.4f)  \t%.4f (%.4f)  \t %d/%d\t\t%.4f (%.4f)\n',j,err_rms_train(end), err_grad_train, err_rms_val(end), err_grad_val, val_fail, val_stop, err_rms_test(end), err_grad_test);
        
        % Stopping condition
        if err_rms_train(end) < stop_err_rms
            fprintf('\nReached stopping criteria: Root mean square error < %.4f\n', stop_err_rms);
            break;
        elseif abs(err_grad_train) < stop_err_grad
            fprintf('\nReached stopping criteria: Error gradient < %.4f\n', stop_err_grad);
            break;
        elseif val_fail >= val_stop
            fprintf('\nValidation test failed %d times (stop to prevent over-training)\n', val_fail);
            break;
        elseif j == num_epoch
            fprintf('\nReached stopping criteria: Max epochs reached\n');
        end
    end
    
    % Use net with best validation error
    net = net_best;

    % Display statistics
    duration = cputime - start_time;
    fprintf('\n')
    fprintf('Duration:         %d min %d sec\n',floor(duration/60), round(mod(duration,60)));
end
