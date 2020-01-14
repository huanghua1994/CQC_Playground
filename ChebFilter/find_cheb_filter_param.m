function [k, g] = find_cheb_filter_param(t0, t1, tau)
% Find parameters k and g for Chebyshev polynomial filtering function
% that amplifies interval [t0, t1] to threshold tau
% Input parameters:
%   t0, t1 : Interval to be amplified, -1 <= t0 < t1 <= 1
%   tau    : Amplifying threshold, p(x) >= tau for all x in (t0, t1)
% Output parameters:
%   k : Degree of Chebyshev polynomial
%   g : Center of the amplifying interval (p(g) = 1)
    if (nargin < 3), tau = 0.8; end
    if (tau < 0.5),  tau = 0.5; end
    if (tau > 1.0),  tau = 0.8; end
    for k = 2 : 50
        g = find_cheb_balance_root(k, t0, t1);
        p_t0 = cheb_filter(k, g, t0);
        p_t1 = cheb_filter(k, g, t1);
        if ((p_t0 < tau) && (p_t1 < tau)), break; end
    end
end

function g = find_cheb_balance_root(k, t0, t1)
% Find parameter g s.t. cheb_filter(k, g, t0) = cheb_filter(k, g, t1)
% Input parameters:
%   k      : Degree of Chebyshev polynomial
%   t0, t1 : Interval to be amplified
% Output parameter:
%   g : Solution to cheb_filter(k, g, t0) = cheb_filter(k, g, t1)
    g  = (t0 + t1) * 0.5;
    g0 = 1.0;
    dg = g0 - g;
    h  = 1e-8;
    while (abs(dg) > 1e-8)
        gpe  = g + h;
        gme  = g - h;
        fc0  = cheb_filter(k, gme, t1) - cheb_filter(k, gme, t0);
        fc1  = cheb_filter(k, gpe, t1) - cheb_filter(k, gpe, t0);
        dfdg = (fc1 - fc0) / (2 * h);
        f  = cheb_filter(k, g, t1) - cheb_filter(k, g, t0);
        g0 = g;
        g  = g0 - f / dfdg;
        dg = g - g0;
    end
end