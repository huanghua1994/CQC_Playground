function [n_rad, n_ang] = NWChem_rad_ang_num(nuc, lvl)
% Get the number of radial and angular grids used in NWChem
% Ref: http://www.nwchem-sw.org/index.php/Release66:Density_Functional_Theory_for_Molecules
% Input parameters:
%   nuc : Nuclear charge
%   lvl : Big number for larger mesh grids, default is 3 (medium)
% Output parameter:
%   n_rad : Number of radial grids
%   n_ang : Number of angular grids

    if (nargin == 1), lvl = 3; end
    tab    = [  10,   18,   36,   54];
    % Period     2,    3,    4,    5        % Level
    n_rads = [  21,   42,   75,   84;  ...  % 1, xcoarse
                35,   70,   95,  104;  ...  % 2, coarse
                49,   88,  112,  123;  ...  % 3, medium
                70,  123,  130,  141;  ...  % 4, fine
               100,  125,  160,  205]; ...  % 5, xfine
    n_angs = [ 194,  194,  194,  194;  ...  % 1, xcoarse
               302,  302,  302,  302;  ...  % 2, coarse
               434,  434,  590,  590;  ...  % 3, medium
               590,  770,  974,  974;  ...  % 4, fine
              1202, 1454, 1454, 1454]; ...  % 5, xfine
    period = 1;
    if (nuc > tab(1)), period = 2; end
    if (nuc > tab(2)), period = 3; end
    if (nuc > tab(3)), period = 4; end
    n_rad = n_rads(lvl, period);
    n_ang = n_angs(lvl, period);
end