function [natom, atom_xyz, bf_coef, bf_alpha, bf_exp, bf_center, bf_nprim] = shell_info_to_bf(nshell, shells)
% Convert shell array into multiple arrays of basis function info
% Input parameters:
%   nshell : Total number of shells
%   shells : Array of shell structures 
% Output parameters:
%   natom     : Number of atoms
%   atom_xyz  : Size natom*3, atom coordinates
%   bf_coef   : Size nbf*max_nprim, coef terms of basis functions
%   bf_alpha  : Size nbf*max_nprim, alpha terms of basis functions
%   bf_exp    : Size nbf*3, polynomial exponents terms of basis functions
%   bf_center : Size nbf*3, center of basis functions
%   bf_nprim  : Size nbf, number of primitive functions in each basis function
    
    natom = max([shells.atom_ind]);
    atom_xyz = zeros(natom, 3);
    for i = 1 : nshell
        ind = shells(i).atom_ind;
        atom_xyz(ind, 1) = shells(i).x;
        atom_xyz(ind, 2) = shells(i).y;
        atom_xyz(ind, 3) = shells(i).z;
    end

    nbf = 0; max_nprim = 0;
    for i = 1 : nshell
        max_nprim = max(max_nprim, shells(i).nprim);
        nbf = nbf + (shells(i).am+2)*(shells(i).am+1)/2;
    end

    bf_coef   = zeros(nbf, max_nprim);
    bf_alpha  = zeros(nbf, max_nprim);
    bf_exp    = zeros(nbf, 3);
    bf_center = zeros(nbf, 3);
    bf_nprim  = zeros(nbf, 1);

    ibf = 1;
    for i = 1 : nshell
        shell_nbf   = (shells(i).am+2)*(shells(i).am+1)/2;
        shell_nprim = shells(i).nprim;
        for j = ibf : ibf + shell_nbf - 1
            bf_coef(j, 1 : shell_nprim) = shells(i).coef(1 : shell_nprim);
            bf_alpha(j, 1 : shell_nprim) = shells(i).alpha(1 : shell_nprim);
            bf_center(j, 1 : 3) = [shells(i).x, shells(i).y, shells(i).z];
            bf_nprim(j) = shell_nprim;
        end
        
        % ??? From DFTfun, initialization_HF.m
        for j = ibf : ibf + shell_nbf - 1
            if (shells(i).am == 0)
                bf_coef(j, :) = bf_coef(j, :) .* (2 * bf_alpha(j, :) / pi).^0.75;
            end
            if (shells(i).am == 1)
                bf_coef(j, :) = bf_coef(j, :) .* (2 * bf_alpha(j, :) / pi).^0.75;
                bf_coef(j, :) = bf_coef(j, :) .* (bf_alpha(j, :) / pi).^0.5 * 2;
            end
            % Rest?
        end

        shell_am = shells(i).am; 
        for xe = shell_am : -1 : 0
            for ye = (shell_am - xe) : -1 : 0
                ze = shell_am - xe - ye;
                bf_exp(ibf, 1) = xe;
                bf_exp(ibf, 2) = ye;
                bf_exp(ibf, 3) = ze;
                ibf = ibf + 1;
            end
        end
    end
end