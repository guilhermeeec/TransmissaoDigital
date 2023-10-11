function [x_adv,predictions] = adversarial_examples(dlnet,mbq,epsilon,alpha,num_iter)

    x_adv = {};
    predictions = [];
    iteration = 0;

    % Generate adversarial images for each mini-batch.
    while hasdata(mbq)

        iteration = iteration +1;
        [dl_x,dl_y] = next(mbq);

        initialization = "zero";

        % Generate adversarial images.
        x_adv_mbq = basic_iterative_method(dlnet,dl_x,dl_y,alpha,epsilon, ...
            num_iter,initialization);

        % Predict the class of the adversarial images.
        dl_y_pred = predict(dlnet,x_adv_mbq);

        x_adv{iteration} = x_adv_mbq;
        predictions = [predictions dl_y_pred];
    end

    % Concatenate.
    x_adv = cat(2,x_adv{:});

end