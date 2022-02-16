function [re_c_X re_f_X] = prodVarCon(c_X, f_X, N)


border = c_X(2) - c_X(1);



re_c_X = c_X.*N;
re_f_X = f_X;
