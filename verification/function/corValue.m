function re_value = corValue(X, Y)

re_value = sum(X .* Y)/(sum(X .* X)^(0.5) * sum(Y .* Y)^(0.5));
