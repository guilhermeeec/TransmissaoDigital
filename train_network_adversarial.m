function dlnet = train_network_adversarial(layers,num_epochs,train_dataset,epsilon,lambda,r,SNR)
    H = readmatrix(train_dataset);
    x_train = H(:,1:2048);
    y_train = H(:,2049:end);
    x_train_ds = arrayDatastore(x_train);
    y_train_ds = arrayDatastore(y_train);
    ds_train = combine(x_train_ds, y_train_ds);

    lgraph = layerGraph(layers);
    dlnet = dlnetwork(lgraph);

    mini_batch_size  = 50;
    learn_rate = 1e-4;
    gradDecay = 0.75;
    sqGradDecay = 0.95;
    %validation_frequency = floor(numel(y_train)/mini_batch_size);
    executionEnvironment = "auto";
    
    alpha = 1.25*epsilon;
    %alpha = epsilon;
    numIter = 1;
    initialization = "random";
    
    mbq = minibatchqueue(ds_train, ...
        'MiniBatchSize',mini_batch_size,...
        'MiniBatchFormat',{'BC','BC'});
    
    iteration = 0;
    start_timestamp = tic;
    test = [];
    velocity = [];
    averageGrad = [];
    averageSqGrad = [];

    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    set(gca, 'YScale', 'log')
    xlabel("Iteration")
    ylabel("Loss")
    grid on

    % Loop over epochs.
    for epoch = 1:num_epochs

        % Shuffle data.
        shuffle(mbq);

        if epoch>50
            learn_rate = 1e-5;
        end
        if epoch>100
            learn_rate = 1e-6;
        end

        % Loop over mini-batches.
        while hasdata(mbq)
            iteration = iteration +1;

            % Read mini-batch of data.
            [dl_x,dl_y] = next(mbq);

            dl_x = basic_iterative_method_new(dlnet,dl_x,dl_y,alpha,epsilon,numIter,initialization,lambda,r,SNR);

            % Evaluate the model gradients, state, and loss.
            [gradients,state,loss] = dlfeval(@model_gradients_new,dlnet,dl_x,dl_y,lambda);
            dlnet.State = state;
            %test = [test; dlnet.State.Value{1}];

            % Update the network parameters using the SGDM optimizer.
            %[dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,learn_rate);

            [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,learn_rate);
            
            % Display the training progress.
            D = duration(0,0,toc(start_timestamp),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,loss)
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end