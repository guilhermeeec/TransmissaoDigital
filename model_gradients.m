function [gradients,state,loss] = model_gradients(net,dl_x,dl_y)

    [dl_y_pred,state] = forward(net,dl_x);          % Passa conjunto pela rede
    
    loss = mse(dl_y_pred,dl_y);                     % Calcula perda em relação ao rótulo
    gradients = dlgradient(loss,net.Learnables);

    loss = double(gather(extractdata(loss)));
end