function pertubations = adversarial_examples_eavesdrop(dlnet,mbq,epsilon,alpha,num_iter)
%[x_adv,predictions]
 
    pertubations = {};
    predictions = [];
    iteration = 0;

    % Generate adversarial images for each mini-batch.
    while hasdata(mbq)

        iteration = iteration +1;
        [dl_x,dl_y] = next(mbq);

        initialization = "zero";

        % Generate adversarial images.
        pertubations_mbq = basic_iterative_method_eavesdrop(dlnet,dl_x,dl_y,alpha,epsilon, ...
            num_iter,initialization);

        pertubations{iteration} = pertubations_mbq;
    end

    % Concatenate.
    pertubations = cat(2,pertubations{:});

end