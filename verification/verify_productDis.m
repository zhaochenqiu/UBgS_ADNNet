clear all
close all
clc

addpath('./function')

rng(0)

num = 1000000;


X = normrnd(10, 2, [num, 1]);
W_t = normrnd(20, 4, [num, 1]);



%X = round(X);
%W_t = round(W_t);



Z_t = X .* W_t;



W = rand(num, 1)*39 + 1;
W = round(W);

Z = X .* W;




border = 1;



left  = min(X);
right = max(X);

left  = round(left - 0.5);
right = round(right + 0.5);

[f_X c_X] = getHist_plus(X, border, left, right);







left  = min([min(W) min(W_t)]);
right = max([max(W) max(W_t)]);

left  = round(left - 0.5);
right = round(right + 0.5);

[f_W   c_W] = getHist_plus(W, border, left, right);
[f_W_t c_W] = getHist_plus(W_t, border, left, right);





left  = min([min(Z) min(Z_t)]);
right = max([max(Z) max(Z_t)]);

left  = round(left - 0.5);
right = round(right + 0.5);

[f_Z   c_Z]     = getHist_plus(Z, border, left, right);
[f_Z_t c_Z]     = getHist_plus(Z_t, border, left, right);







learning_rate = 0.1;

% epoch = 1;



figure


set (gcf,'Position',[10 180 1200 500], 'Color',[1 1 1])
for epoch = 1:3000

    df_W = abs(f_W - f_W);


    df_Z = f_Z - f_Z_t;



    for i = 1:max(size(c_W))
        w = c_W(i);

        d_cum = 0;
        if w ~= 0
            for j = 1:max(size(c_Z))
                z = c_Z(j);

                dfz = df_Z(j);

                x = round( (z/w)*(1/border) )*border;
                pos = find( abs(c_X - x) < border*0.5 );

                if min(size(pos)) > 0
                    d_cum = d_cum + f_X(pos)*dfz*(1/abs(w));
                end
            end
        end

        df_W(i) = d_cum;
    end


    f_W = f_W - df_W*learning_rate;
    f_W(f_W < 0) = 0;


    value = sum(f_W);
    f_W = f_W ./ value;


    W = getRandbyF_int(f_W, c_W, num);
    Z = X .* W;
    [f_Z   c_Z]     = getHist_plus(Z, border, left, right);


    p_value = sum(f_Z .* f_Z_t)/(sum(f_Z .* f_Z)^(0.5) * sum(f_Z_t .* f_Z_t)^(0.5));
    p_value

    subplot(2, 3, 1)
    bar(c_X, f_X, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
    title('X: Z = X*W')

    subplot(2, 3, 2)
    bar(c_W, f_W, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
    title('W: Z = X*W')


    subplot(2, 3, 3)
    bar(c_W, f_W_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
    title('W: groundtruth')

    subplot(2, 3, 4)
    bar(c_Z, f_Z, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
    title('output distribution')

    subplot(2, 3, 5)
    bar(c_Z, f_Z_t, 1, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [0.5 0.5 0.5])
    title('target distribution')

    subplot(2, 3, 6)
    hold on
    title('correlation value')
    plot(epoch, p_value,'.:', 'Color', [0.5 0.5 0.5])


    drawnow

%     input('pause')
end


 % sum(f_t)
