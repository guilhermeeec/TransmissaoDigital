function gradient = model_gradients_input_new(dlnet,dl_x,y,lambda)

    [dl_y_pred] = forward(dlnet,dl_x);

    loss = mse(dl_y_pred,y) - lambda*mse(dl_x,y);
    
    gradient = dlgradient(loss,dl_x);

end