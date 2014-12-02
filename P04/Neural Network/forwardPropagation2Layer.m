function [ Ys, As1 ] = forwardPropagation2Layer(Thetas1, Thetas2, Xs)

    % applies forward propagation to a 2 layer neural network. 
    % completey vectorized for speed
    
    % add bias node 1 to Xs
    Xs_size = size(Xs);
    Xs = [ones(Xs_size(1), 1) Xs];
    
    % propagate to hidden layer, calculate all z values
    Zs1 = Xs * Thetas1;
    
    % calculate activation values for all hidden layer nodes
    As1 = sigmoid(Zs1);
    
    % add bias node 1 to As1 nodes
    As1_size = size(As1);
    As1 = [ones(As1_size(1),1) As1];
    
    % propagate to second layer
    Zs2 = As1 * Thetas2;
    
    % calculate activation values for output node
    Ys = sigmoid(Zs2);
    
end

