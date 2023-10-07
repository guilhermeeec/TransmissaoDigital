function [up,pilot_symbols] = generate_transmitter_signal(N,K)
    pilot_symbols = (rand(1,N)+1i*rand(1,N))';  % Símbolos combinados para estimação de canal
    x = sqrt(N)*ifft(pilot_symbols);                 

    A_cp = [zeros(K,N-K) eye(K); eye(N)];       % Matriz de adição de prefixo cíclico
    up = A_cp*x;
end