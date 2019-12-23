function rad_n_ang = NWChem_prune_grid(nuc, n_ang, n_rad, rads)
% Prune grids using NWChem scheme
% Ref: 
%   1. https://github.com/pyscf/pyscf/blob/master/pyscf/dft/gen_grid.py
%   2. https://github.com/pyscf/pyscf/blob/master/pyscf/data/radii.py
% Input parameters:
%   nuc   : Nuclear charge
%   n_ang : Maximum number of angular grids
%   n_rad : Number of radial grids
%   rads  : Array, size n_rad, radial grid coordinates
% Output parameter:
%   rad_n_ang : Array, size n_rad, number of angular grids for each radial grid
    
    Leb_ngrid = [   1,    6,   14,   26,   38,   50,   74,   86,  110,  146, ...
                  170,  194,  230,  266,  302,  350,  434,  590,  770,  974, ...
                 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, ...
                 4802, 5294, 5810];
    alphas = [0.25, 0.5, 1.0, 4.5; ...
               1/6, 0.5, 0.9, 3.5; ...
               0.1, 0.4, 0.8, 2.5];
    BOHR = 0.52917721092;
    RADII_BRAGG = (1 / BOHR) .* [ ...
        0.35,                                     1.40,             ... % 1s
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             ... % 2s2p
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             ... % 3s3p
        2.20, 1.80,                                                 ... % 4s
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, ... % 3d
                    1.30, 1.25, 1.15, 1.15, 1.15, 1.90,             ... % 4p
        2.35, 2.00,                                                 ... % 5s
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, ... % 4d
                    1.55, 1.45, 1.45, 1.40, 1.40, 2.10,             ... % 5p
        2.60, 2.15,                                                 ... % 6s
        1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   ... % La, Ce-Eu
        1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             ... % Gd, Tb-Lu
              1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, ... % 5d
                    1.90, 1.80, 1.60, 1.90, 1.45, 2.10,             ... % 6p
        1.80, 2.15,                                                 ... % 7s
        1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,                   ...
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,                   ...
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, ...
                    1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             ...
        1.75, 1.75,                                                 ...
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75  ...
    ];
    
    rad_n_ang = zeros(n_rad, 1);
    if (n_ang < 50)
        rad_n_ang(:) = n_ang;
        return;
    end
    if (n_ang == 50)
        leb_l = [5, 6, 6, 6, 5];
    end
    if (n_ang > 50)
        for idx = 7 : 32
            if (n_ang == Leb_ngrid(idx)), break; end
        end
        leb_l = [5, 7, idx-1, idx, idx-1];
    end
    r_atom = RADII_BRAGG(nuc);
    for i = 1 : n_rad
        rad_i_scale = rads(i) / r_atom;
        if (nuc >  10), place = sum(rad_i_scale > alphas(3, :)); end
        if (nuc <= 10), place = sum(rad_i_scale > alphas(2, :)); end
        if (nuc <=  2), place = sum(rad_i_scale > alphas(1, :)); end
        rad_n_ang(i) = Leb_ngrid(leb_l(place + 1));
    end
end