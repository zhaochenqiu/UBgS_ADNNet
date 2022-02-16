function [c_Z f_Z] = productDis_plus(c_X, f_X, c_Y, f_Y)


border_X = c_X(2) - c_X(1);
border_Y = c_Y(2) - c_Y(1);

border = min([border_X border_Y]);



left = min(c_X);
left = min([left min(c_Y)]);
left = min([left min(c_Y)*min(c_X)]);
left = min([left min(c_Y)*max(c_X)]);
left = min([left max(c_Y)*min(c_X)]);
left = min([left max(c_Y)*max(c_X)]);

left = round(left/border - 0.5)*border;



right = max(c_X);
right = max([right max(c_Y)]);
right = max([right min(c_Y)*min(c_X)]);
right = max([right min(c_Y)*max(c_X)]);
right = max([right max(c_Y)*min(c_X)]);
right = max([right max(c_Y)*max(c_X)]);

right = round(right/border + 0.5)*border;



c_Z = left:border:right;
f_Z = abs(c_Z - c_Z);



for i = 1:max(size(c_Z))
    z = c_Z(i);


    idx_nonzero = c_X ~= 0;
    y = round( (z ./ c_X(idx_nonzero) )*(1/border) );

    offset_y = round(min(c_Y/border));

    pos = y - offset_y + 1;


    min_pos = 0;
    max_pos = max(size(c_Y)) + 1;

    idx_exist = ((pos > min_pos) & (pos < max_pos));
    pos = pos(idx_exist);



    x = c_X(idx_nonzero);
    x = x(idx_exist);


    p_x = f_X(idx_nonzero);
    p_x = p_x(idx_exist);



%    p_all = p_all + sum(p_x .* f_Y(pos) .* abs((1 ./ x)));

    f_Z(i) = sum(p_x .* f_Y(pos) .* abs((1 ./ x)));


    deltaX = 2*border;
    x_l = 0 - 1*border;
    x_r = 0 + 1*border;

    y_l = z/x_l;
    y_r = z/x_r;

    y_l = round(y_l / border)*border;
    y_r = round(y_r / border)*border;

    idx_l = c_Y == y_l;
    idx_r = c_Y == y_r;
    p_Y = (f_Y(idx_l) + f_Y(idx_r))/deltaX;
%    p_Y = abs(p_Y)
    p_Y = p_Y*2.26;


    idx_0 = c_X == 0;

    if min(size(p_Y)) < 1
        p_Y = 0;
    end

    value = f_X(idx_0) * p_Y;


    if min(size(value)) < 1
        value = 0;
    end

    if value ~= 0
        z
        value
        input('pause')
    end

    f_Z(i) = f_Z(i) + value;

end
