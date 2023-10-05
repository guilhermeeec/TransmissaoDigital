function mse = my_mse(signal_1, signal_2)
    mse = mean(abs(signal_1-signal_2).^2);
end