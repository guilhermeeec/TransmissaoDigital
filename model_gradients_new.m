function [gradients,state,loss] = model_gradients_new(net,dl_x,dl_y,lambda)

    [dl_y_pred,state] = forward(net,dl_x);          % Passa conjunto pela rede
    
    loss = mse(dl_y_pred,dl_y) - lambda*mse(dl_x,dl_y);      
    
    gradients = dlgradient(loss,net.Learnables);

    loss = double(gather(extractdata(loss)));
end