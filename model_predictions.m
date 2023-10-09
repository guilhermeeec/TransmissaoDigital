function predictions = model_predictions(net,mbq)

    predictions = [];

    while hasdata(mbq)

        dl_x_test = next(mbq);
        dl_y_pred = predict(net,dl_x_test);

        predictions = [predictions; dl_y_pred];
    end

end