function result = signal_power(signal)
    result = mean(abs(signal).^2);
end