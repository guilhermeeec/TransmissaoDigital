function result = add_noise(signal,SNR)
    l = length(signal);
    snr = 10^(SNR/10);
    Psignal = mean(abs(signal).^2); 
    Pnoise = Psignal / snr;
    noise = wgn(l,1,10*log10(Pnoise/2)) + 1i*wgn(l,1,10*log10(Pnoise/2));
    result = signal + noise;
end