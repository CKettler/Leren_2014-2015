function [ column ] = normalizeColumn( column )
    % normalizes a column

    min_val = min(column);
    max_val = max(column);
    mean_val = mean(column);
    range = max_val - min_val;
    if (range == 0)
        range = 1;
    end
    
    % x = (x - mean(Col))/(max(Col) - min(Col))
    column = 1/range * (column - mean_val);

end

