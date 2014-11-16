figure('name','Plotting XOR');
axis([0 1 0 1]);
xlabel('X1');
ylabel('X2');
title('Plotting XOR');
hold on;
scatter([0 1],[0 1], 'MarkerFaceColor', 'r');
scatter([1 0],[0 1], 'MarkerFaceColor', 'g');
plot([0.5 0.5], [0 1]);
plot([0 1], [0.5 0.5]);
hold off;

