function retrieved_symbols = generate_received_signal(z,N,K)
    R_cp = [zeros(N,K) eye(N)];                 % Matriz de remoção de prefixo cíclico
    y = R_cp*z;
    
    retrieved_symbols = fft(y)/sqrt(N);
end