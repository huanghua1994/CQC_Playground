function [rad_r, rad_w] = my_cheb2_becke(n, rm)
% Ref: [JCP 88, 2547], doi: 10.1063/1.454033
% Input parameters:
%   n  : Number of points on the radial direction
%   rm : A parameter mentioned in Ref.
% Output parameters:
%   rad_r : Size n-1, radial direction integral points
%   rad_w : Size n-1, radial direction integral point weights

    rad_r = zeros(n, 1);
    rad_w = zeros(n, 1);
    m = n - 1;
    for i = 0 : m
        radx = cos(i * pi / m);
        radr = rm * (1 + radx) / (1 - radx);
        % radw formula comes from: http://sobereva.com/69, or 
        % multiwfn v3.7 fuzzy.f90, line 449
        radw = (2 * pi / m) * (rm^3) * (1 + radx).^2.5 / ((1 - radx).^3.5);
        rad_r(i+1) = radr;
        rad_w(i+1) = radw;
    end
    % rad_w(1) = inf, discard it
    rad_r = rad_r(end : -1 : 2);
    rad_w = rad_w(end : -1 : 2);
end