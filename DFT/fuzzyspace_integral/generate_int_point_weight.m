function [ip, ipw] = generate_int_point_weight(atom_xyz, atom_num)
% Get numerical integral points and their weights for XC integral
% Ref: [JCP 88, 2547], doi: 10.1063/1.454033, and http://sobereva.com/69
% Input parameter:
%   atom_xyz : Atom coordinates
%   atom_num : Atom number (H:1, He:2, Li:3, ...)
% Output parameters:
%   ip  : Numerical integral points of all atoms
%   ipw : Weights of numerical integral points

    rm = 1;         % A parameter used in my_cheb2_becke
    max_rad = 65;   % Maximum radial direction points
    max_ang = 302;  % Maximum sphere points
    
    natom = size(atom_xyz, 1);
    ip    = zeros(max_rad * max_ang * natom, 3);
    ipw   = zeros(max_rad * max_ang * natom, 1);
    dist  = squareform(pdist(atom_xyz));
    cnt   = 0;
    for iatom = 1 : natom
        % (1) Prune grid points according to atom type
        n_ang = max_ang;   
        n_rad = max_rad;
        if (atom_num(iatom) <= 10), n_rad = 50; end
        if (atom_num(iatom) <= 2),  n_rad = 35; end
        [rad_r, rad_w] = cheb2_becke(n_rad + 2, rm);
        n_rad = length(rad_r);
        rad_n_ang = NWChem_prune_grid(atom_num(iatom), n_ang, n_rad, rad_r);
        nintp_atom = sum(rad_n_ang);
        
        % (2) Generate Lebedev points & weights and combine it
        %     with radial direction points & weights
        ip_atom  = zeros(nintp_atom, 3);
        ipw_atom = zeros(nintp_atom, 1);
        spos = 1;
        for j = 1 : n_rad
            Lebedev_pw = getLebedevSphere(rad_n_ang(j));
            epos = spos + rad_n_ang(j) - 1;
            ip_atom(spos : epos, 1) = Lebedev_pw.x * rad_r(j);
            ip_atom(spos : epos, 2) = Lebedev_pw.y * rad_r(j);
            ip_atom(spos : epos, 3) = Lebedev_pw.z * rad_r(j);
            ipw_atom(spos : epos)   = Lebedev_pw.w * rad_w(j);
            spos = epos + 1;
        end
        
        % (3) Calculate the mask tensor and the actual weights
        % W_mat(i, j, k): fuzzy weight of integral point i to atom pair (j, k)
        W_mat = zeros(nintp_atom, natom, natom);
        % Shift the integral point to atom
        rnowx = ip_atom(:, 1) + atom_xyz(iatom, 1);
        rnowy = ip_atom(:, 2) + atom_xyz(iatom, 2);
        rnowz = ip_atom(:, 3) + atom_xyz(iatom, 3);
        dip = zeros(nintp_atom, natom);
        for jatom = 1 : natom 
            dxj = rnowx - atom_xyz(jatom, 1);
            dyj = rnowy - atom_xyz(jatom, 2);
            dzj = rnowz - atom_xyz(jatom, 3);
            dip(:, jatom) = sqrt(dxj.^2 + dyj.^2 + dzj.^2);
        end
        % Enumerate each atom pair (j, k)
        for jatom = 1 : natom 
            for katom = 1 : natom
                if katom ~= jatom
                    smu = (dip(:, jatom) - dip(:, katom)) / dist(jatom, katom);
                    
                    % s(d(i,j)) = 0.5 * (1 - p(p(p(d(i,j)))))
                    for k = 1 : 3
                        smu = 1.5 * smu - 0.5 * smu.^3;
                    end
                    W_mat(:, jatom, katom) = (0.5 * (1 - smu));
                else
                    W_mat(:, jatom, katom) = 1;
                end
            end
        end
        
        % (4) Calculate the final integral weights
        % \prod_{k} W_mat(:, j, k) is the actual weight of integral points
        % belonging to atom k. Normalizing it gives us the fuzzy weight.
        pvec = ones(nintp_atom, natom);
        for i = 1 : natom  
            for j = 1 : natom     
                pvec(:, i) = pvec(:, i) .* W_mat(:, i, j);
            end  
        end
        sum_pvec = sum(pvec, 2);
        % Copy the final integral points & weights to the output matrix
        sidx = cnt + 1;
        eidx = cnt + nintp_atom;
        ip(sidx : eidx, 1) = rnowx;
        ip(sidx : eidx, 2) = rnowy;
        ip(sidx : eidx, 3) = rnowz;
        ipw(sidx : eidx) = ipw_atom .* pvec(:, iatom) ./ sum_pvec;
        cnt = cnt + nintp_atom;
    end
    
    ip  = ip(1 : cnt, :);
    ipw = ipw(1 : cnt);
end