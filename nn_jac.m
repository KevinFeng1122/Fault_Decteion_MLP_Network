function jac = nn_jac(net,input)
    % Initialize Jacobian
    w_vect = getx(net);%getwb
    n = numel(w_vect);
    m = numel(sim(net,input));
    jac = zeros(m,n);
    
    % Calculate Jacobian using complex step differentiation
    h = n*eps;
    for k = 1:n
        w_vect_alt      = w_vect;
        w_vect_alt(k)   = w_vect_alt(k) + h*1i;
        net_alt         = setx(net,w_vect_alt);%setwb
        alt_out         = imag(sim(net_alt,input))/h;
        jac(:,k)        = reshape(alt_out',m,1);
    end
end