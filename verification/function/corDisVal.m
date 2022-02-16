function re_value = corDisVal(c_X, f_X, c_Y, f_Y)


left    = max([min(c_X) min(c_Y)]);
right   = min([max(c_X) max(c_Y)]);

border = c_X(2) - c_X(1);


pos_l = find( abs(c_X - left)  < 0.5*border);
pos_r = find( abs(c_X - right) < 0.5*border);

f_X = f_X(pos_l:pos_r);



pos_l = find( abs(c_Y - left)  < 0.5*border);
pos_r = find( abs(c_Y - right) < 0.5*border);

f_Y = f_Y(pos_l:pos_r);


re_value = sum(f_Y .* f_X)/(sum(f_Y .* f_Y)^(0.5) * sum(f_X .* f_X)^(0.5));


