function [X,T] = preprocess_mini_batch(XCell,TCell)
    % Concatenate.
    X = cat(1,XCell{1:end});
    X = single(X);

    % Extract label data from the cell and concatenate.
    T = cat(1,TCell{1:end});
end