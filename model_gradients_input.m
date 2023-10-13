function gradient = model_gradients_input(dlnet,dl_x,y)

    [dl_y_pred] = forward(dlnet,dl_x);

    loss = mse(dl_y_pred,y);
        
    gradient = dlgradient(loss,dl_x);

end