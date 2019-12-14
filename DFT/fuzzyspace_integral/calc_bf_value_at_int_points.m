function rho = calc_bf_value_at_int_points(int_points, atom_xyz, nbf, bf_coef, bf_alpha, bf_exp, bf_center, bf_nprim)
% Evaluate the values of basis function at integral points
% Input parameters:
%   int_points : Integral point coordinates
%   atom_xyz   : Atom coordinates
%   nbf        : Total number of basis functions
%   bf_coef    : coef  terms  of basis functions
%   bf_alpha   : alpha terms  of basis functions
%   bf_exp     : Polynomial exponents terms of basis functions
%   bf_center  : Center of basis functions
%   bf_nprim   : Number of primitive functions in each basis function
% Output parameter:
%   rho(:, i, k): i-th basis function value at k-th atom integral points

    natom = size(atom_xyz, 1);
    nintp = size(int_points, 1);
    rho   = zeros(nintp, nbf, natom);
    for k = 1 : natom
        ipx = int_points(:, 1) + atom_xyz(k, 1);
        ipy = int_points(:, 2) + atom_xyz(k, 2);
        ipz = int_points(:, 3) + atom_xyz(k, 3);
        for i = 1 : nbf
            bfx  = bf_center(i, 1);
            bfy  = bf_center(i, 2);
            bfz  = bf_center(i, 3);
            bfxe = bf_exp(i, 1);
            bfye = bf_exp(i, 2);
            bfze = bf_exp(i, 3);
            
            poly = (ipx-bfx).^bfxe .* (ipy-bfy).^bfye .* (ipz-bfz).^bfze;
            dx2  = (ipx-bfx).^2;
            dy2  = (ipy-bfy).^2;
            dz2  = (ipz-bfz).^2;
            d2   = dx2 + dy2 + dz2;
            
            rho_i_k = zeros(nintp, 1);
            for p = 1 : bf_nprim(i)
                rho_i_k = rho_i_k + (bf_coef(i, p) .* poly) .* exp(-bf_alpha(i, p) * d2);
            end
            rho(:, i, k) = rho_i_k;
        end
    end
end