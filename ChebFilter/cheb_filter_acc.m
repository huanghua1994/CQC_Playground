function [k, g, min_ev1, max_occ1, min_uocc1, max_ev1] = cheb_filter_acc(min_ev, max_occ, min_uocc, max_ev)
% Use Chebyshev polynomial filter to amplify the gap [max_occ, min_uocc]
% Input parameter:
%   min_ev   : Minimal eigenvalue
%   max_occ  : Maximal eigenvalue corresponding to the occupied orbitals
%   min_uocc : Minimal eigenvalue corresponding to the unoccupied orbitals
%   max_ev   : Maximal eigenvalue
% Output parameter:
%   k, g      : Chebyshev polynomial filter p(x) parameters
%   min_ev1   : p(min_ev)
%   max_occ1  : p(max_occ)
%   min_uocc1 : p(min_uocc)
%   max_ev1   : p(max_ev)
    
    % Map min_ev to 1, max_ev to -1 for Chebyshev polynomial filtering
    lambda    = 2 / (max_ev - min_ev);
    min_ev1   = 1;
    max_ev1   = -1;
    max_occ1  = 1 - lambda * (max_occ  - min_ev);
    min_uocc1 = 1 - lambda * (min_uocc - min_ev);
    [k, g] = find_cheb_filter_param(max_occ1, min_ev1, 0.5);
    x  = linspace(-1, 1, 201);
    cx = cheb_filter(k, g, x);
    min_ev1   = min(cx);
    max_ev1   = max(cx);
    max_occ1  = cheb_filter(k, g, max_occ1);
    min_uocc1 = cheb_filter(k, g, min_uocc1);
end