function x_adv = basic_iterative_method_new(dlnet,dl_x,dl_y,alpha,epsilon,num_iter,initialization,lambda,r,SNR)

    if initialization == "zero"
        delta = zeros(size(dl_x),'like',dl_x);
    else
        delta = epsilon*(2*rand(size(dl_x),'like',dl_x) - 1);
    end
    
    epsilon = epsilon * (1-r)^(SNR/5-1);

    for i = 1:num_iter

        % Apply adversarial perturbations to the data.
        gradient = dlfeval(@model_gradients_input_new,dlnet,dl_x+delta,dl_y,lambda);
        delta = delta + alpha*sign(gradient);
        %delta = delta + epsilon*sign(gradient);
        delta(delta > epsilon) = epsilon;
        delta(delta < -epsilon) = -epsilon;
    end

    x_adv = dl_x + delta;

end