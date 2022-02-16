function re_X = getRandbyF_int(f_X, c_X, num)

re_X = [];

for i = 1:max(size(c_X))
    x = c_X(i);
    p_x = f_X(i);

    cnt = round(p_x*num);

    re_X = [re_X; zeros(cnt, 1) + x];
end


idx = randperm(max(size(re_X)));
re_X = re_X(idx);




num_t = max(size(idx));

if num_t > num
    re_X = re_X(1:num);
end

if num_t < num
    re_X = [re_X; re_X(1:(num - num_t))];
end
