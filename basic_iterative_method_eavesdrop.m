function delta = basic_iterative_method_eavesdrop(dlnet,dl_x,dl_y,alpha,epsilon,num_iter,initialization)

    % Initialize the perturbation.
    if initialization == "zero"
        delta = zeros(size(dl_x),'like',dl_x);
    else
        delta = epsilon*(2*rand(size(dl_x),'like',dl_x) - 1);
    end

    for i = 1:num_iter

        % Apply adversarial perturbations to the data.
        gradient = dlfeval(@model_gradients_input,dlnet,dl_x+delta,dl_y);
        delta = delta + alpha*sign(gradient);
        delta(delta > epsilon) = epsilon;
        delta(delta < -epsilon) = -epsilon;
    end

end