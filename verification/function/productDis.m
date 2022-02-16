function [c_Z f_Z] = productDis(c_X, f_X, c_Y, f_Y)


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
    p_all = 0;

    for j = 1:max(size(c_X))
        x = c_X(j);

        if x ~= 0
            p_x = f_X(j);

            y = round( (z/x)*(1/border) )*border;
            pos = find( abs(c_Y - y) < border*0.5 );

            if min(size(pos)) > 0
                p_all = p_all + p_x*f_Y(pos)*(1/abs(x));
            end
        end
    end

    f_Z(i) = p_all;

    [i max(size(c_Z))]
end


