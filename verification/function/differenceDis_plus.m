function [c_Z f_Z] = differenceDis_plus(c_X, f_X, c_Y, f_Y)


border_X = c_X(2) - c_X(1);
border_Y = c_Y(2) - c_Y(1);

border = min([border_X border_Y]);



left = min(c_X) - max(c_Y);
left = round(left/border - 0.5)*border;


right = max(c_X) - min(c_Y);
right = round(right/border + 0.5)*border;


c_Z = left:border:right;
f_Z = abs(c_Z - c_Z);



for i = 1:max(size(c_Z))
    z = c_Z(i);

    y = round( (c_X - z)*(1/border) );
    offset_y = round( min(c_Y/border)  );

    pos = y - offset_y + 1;

    min_pos = 0;
    max_pos = max(size(c_Y)) + 1;

    idx_exist = ((pos > min_pos) & (pos < max_pos));
    pos = pos(idx_exist);


    p_x = f_X(idx_exist);

    f_Z(i) = sum( p_x .* f_Y(pos) );

    [i max(size(c_Z))]


%    input('pause')
%    z
%    f_Z(i)
%    input('pause')
end
