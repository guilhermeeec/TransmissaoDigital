function result = add_noise_from_power(signal,Pnoise)
    l = length(signal);
    noise = wgn(l,1,10*log10(Pnoise/2)) + 1i*wgn(l,1,10*log10(Pnoise/2));
    result = signal + noise;
end