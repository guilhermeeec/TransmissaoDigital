function gradient = model_gradients_input(dlnet,dl_x,y)

    %y = dlarray(y,'CB');

    [dl_y_pred] = forward(dlnet,dl_x);

    loss = mse(dl_y_pred,y);
    %loss = immse(double(extractdata(dl_y_pred)),double(extractdata(y)));                     

    
    gradient = dlgradient(loss,dl_x);

end