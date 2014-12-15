function [ ps ] = run_gauss( xs, covs, means, classes)
    ps = cell(size(classes, 1), 1);
    for c=classes
        cov_m = covs{c};
        d = size(cov_m, 1);
        
        cov_m = cov_m + eye(d);
        
        mx = (xs - means{c})';
        a = (2*pi)^(1/d) * det(cov_m)^(1/2);
        b = 1/2 .* mx' * (cov_m \ mx);
        ps{c} = 1/a * exp(-b);
    end
end

