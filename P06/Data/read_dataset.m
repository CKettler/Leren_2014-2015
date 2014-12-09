function [ Xs, Cs ] = read_dataset( dataset )
    % reads dataset with name and returns feature vector and classes vector

    Xs = dataset(:,(1:end - 1));
    Cs = dataset(:,end);
end

