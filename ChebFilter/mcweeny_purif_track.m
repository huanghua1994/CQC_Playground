function x = mcweeny_purif_track(min_ev, max_occ, min_uocc, max_ev)
% Generate the track of McWeeny purification x_{i+1} = 3 * x_i^2 - 2 * x_i^3 
% where x_0 is the mapped maximal eigenvalue corresponding to the occupied orbitals
% Input parameter:
%   min_ev   : Minimal eigenvalue
%   max_occ  : Maximal eigenvalue corresponding to the occupied orbitals
%   min_uocc : Minimal eigenvalue corresponding to the unoccupied orbitals
%   max_ev   : Maximal eigenvalue
% Output parameter:
%   x : Vector, [x0, x1, ...]

    % Map mu to 0.5, min_ev to 1 or max_ev to 0 for McWeeny purification
    mu = (max_occ + min_uocc) * 0.5;
    l  = min(1 / (mu - min_ev), 1 / (max_ev - mu));
    x0 = 0.5 * l * (mu - max_occ) + 0.5;

    max_iter = 200;
    x = zeros(max_iter, 1);
    i = 1;
    x(i) = x0;
    while (i < max_iter)
        x(i+1) = 3 * x(i) * x(i) - 2 * x(i) * x(i) * x(i);
        if (x(i+1) + 1e-11 > 1), break; end
        i = i + 1;
    end
    x = x(1 : i+1);
end