function [re_Y re_X] = getHist(data, border)

if nargin == 1
    borer == 1;
end


re_X = round(min(data) - 0.5):border:round(max(data) + 0.5);
re_Y = hist(data, re_X) ./ max(size(data));

