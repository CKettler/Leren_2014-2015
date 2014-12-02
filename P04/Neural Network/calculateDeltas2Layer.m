function [ DeltasL, Deltas2 ] = calculateDeltas2Layer( OutputFinal, Ys, ...
                                    OutputHidden, Thetas2, class)
    % calculates deltas (error) for a 2 layer network.
    % input - final outputs as vector
    % Y - target outputs
    % output of hidden layer
    % thetas from hidden layer to output layer
    
    Ynew = Ys;
    Ynew ( Ynew ~= class ) = 0;
    Ynew ( Ynew == class ) = 1;
    
    % calculate deltas for output layer
    DeltasL = OutputFinal - Ynew;
    
    H = (OutputHidden' .* (1 - OutputHidden'));
    H(1) = 1;  % bias term sets first entry to 0. hence we set it to 1 to avoid it cancelling out
    M = Thetas2 .* H;
    Deltas2 = DeltasL * M;
    
end

