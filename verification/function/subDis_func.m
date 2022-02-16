function [re_c_Z re_f_Z] = subDis_func(c_X_, f_X_, c_Y_, f_Y_, c_X, f_X, c_Z)



re_c_Z = c_Z;
re_f_Z = abs(re_c_Z - re_c_Z);
N = 20;

border = 0.1;


for i = 1:max(size(c_Z))
    z = re_c_Z(i);
    p_all = 0;


    for j = 1:max(size(c_Y_))
        y = c_Y_(j);

        if y ~= 0
            p_y = f_Y_(j);

            x = (z + y)*( N/y ) - N;
            x = round(x / border)*border;

            pos = find( abs(c_X - x) < border*0.5 );

            if min(size(pos)) > 0
                p_all = p_all + p_y*p_y*f_X(pos);
            end
        else
            if z == 0
                p_all = p_all + 1*p_y*p_y;
            end
        end
    end


    re_f_Z(i) = p_all;

    [i max(size(re_c_Z))]
end
