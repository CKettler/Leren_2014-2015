% Examples used from:
% https://chrisjmccormick.wordpress.com/2014/08/04/gaussian-mixture-models-tutorial-and-matlab-code/

xs = [1, 2, 3, 3, 4, 5, 5, 7, 10, 11, 13, 14, 15, 17, 20, 21];
mus = [1, 3, 8];
sigmas = [2 2 2];
c1 = [1, 2];
c2 = [3,3,4,5,5];
c3 = [7,10,11,13,14,15,17,20,21];
cprob = [length(c1) / length(xs), length(c2) / length(xs), length(c3) / length(xs)];

E = zeros(length(mus), length(xs));
g = zeros(1, length(mus));

for k = 1:2
    % Calculate expectations
    for i = 1:length(xs)
        for j = 1:length(mus)
            % Gaussian prob
            g(j) = (1 / (sigmas(j) * sqrt(2 * pi))) * exp((-(xs(i) - mus(j))^2) / ...
                                                   (2 * sigmas(j)^2));
            E(j, i) = g(j) * cprob(j);
        end
        sum_prob = sum(E(:,i));
        % Weighted expectation
        for j = 1:length(mus)
            E(j, i) = E(j, i) / sum_prob;
        end
    end

    % Calculate maximizations
    phis = zeros(1, length(mus));
    for j = 1:length(mus)
        % Get mean of expectations for each cluster
        phis(j) = mean(E(j,:));

        % Calc new mean
        mus(j) = E(j,:) * xs';
        mus(j) = mus(j) ./ sum(E(j,:));

        % Get variance
        var = E(j,:) * (xs - mus(j)).^2';
        var = var ./ sum(E(j,:));

        % Calculate new sigmas
        sigmas(j) = sqrt(var);
    end
    E
    mus
    sigmas
end