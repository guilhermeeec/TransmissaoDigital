function [gradients,state,loss] = model_gradients(net,dl_x,dl_y)

    [dl_y_pred,state] = forward(net,dl_x);          % Passa conjunto pela rede
    
    loss = mse(dl_y_pred,dl_y);                     % Calcula perda em relação ao rótulo
    
    %loss = single(dlarray(immse(double(extractdata(dl_y_pred)),double(extractdata(dl_y)))));                     
    %power_dl_y_pred = sqrt(dl_y_pred(1:1024,:).^2 + dl_y_pred(1025:2048,:).^2)
    %power_dl_y = sqrt(dl_y(1:1024,:).^2 + dl_y(1025:2048,:).^2)
    
    % dl_y = dl_y(1:1024,:) + dl_y(1025:2048,:)*1i
    % dl_y_pred = dl_y_pred(1:1024,:) + dl_y_pred(1025:2048,:)*1i
    % loss = mse(dl_y_pred,dl_y);                     % Calcula perda em relação ao rótulo

    
    gradients = dlgradient(loss,net.Learnables);

    loss = double(gather(extractdata(loss)));
end