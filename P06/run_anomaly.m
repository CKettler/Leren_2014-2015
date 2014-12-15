function [ p ] = run_anomaly( xs, cov_m, means)
    
    d = size(cov_m, 1);

    cov_m = cov_m + eye(d);

    mx = (xs - means)';
    a = (2*pi)^(1/d) * det(cov_m)^(1/2);
    b = 1/2 .* mx' * (cov_m \ mx);
    p = 1/a * exp(-b);

end

