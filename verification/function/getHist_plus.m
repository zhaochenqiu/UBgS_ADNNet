function [re_Y re_X] = getHist(data, border, left, right)

if nargin == 1
    borer == 1;
    left  = round(min(data)/border - 0.49999)*border;
    right = round(max(data)/border + 0.49999)*border;
end

if nargin == 2
    left  = round(min(data)/border - 0.49999)*border;
    right = round(max(data)/border + 0.49999)*border;
end


% re_X = round(min(data) - 0.5):border:round(max(data) + 0.5);
re_X = left:border:right;
re_Y = hist(data, re_X) ./ max(size(data));

