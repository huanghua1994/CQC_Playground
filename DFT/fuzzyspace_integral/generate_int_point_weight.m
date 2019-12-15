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
    max_sph = 302;  % Maximum sphere points
    
    natom = size(atom_xyz, 1);
    ip    = zeros(max_rad * max_sph * natom, 3);
    ipw   = zeros(max_rad * max_sph * natom, 1);
    dist  = squareform(pdist(atom_xyz));
    cnt   = 0;
    for iatom = 1 : natom
        % (1) Prune grid points according to atom type
        n_sph = max_sph;   % Do not prune sphere points yet
        n_rad = max_rad;
        if (atom_num(iatom) <= 10), n_rad = 50; end
        if (atom_num(iatom) <= 2),  n_rad = 35; end
        
        % (2) Generate Lebedev points & weights and combine it
        %     with radial direction points & weights
        Lebedev_pw = getLebedevSphere(n_sph);
        [rad_r, rad_w] = cheb2_becke(n_rad, rm);
        n_radcut  = sum(rad_r < 10); 
        ip_atom   = zeros(n_radcut * n_sph, 3);
        ipw_atom  = zeros(n_radcut * n_sph, 1);
        Lebedev_p = [Lebedev_pw.x, Lebedev_pw.y, Lebedev_pw.z];
        for ir = 1 : n_radcut
            spos = (ir-1) * n_sph + 1;
            epos = ir * n_sph;
            ip_atom(spos : epos, 1 : 3) = Lebedev_p * rad_r(ir);
            ipw_atom(spos : epos) = Lebedev_pw.w * rad_w(ir);
        end
        nintp_atom = n_radcut * n_sph;
        
        % (3) Calculate the mask tensor and the actual weights
        % W_mat(i, j, k): fuzzy weight of integral point i to atom pair (j, k)
        % W_mat(i, j, k): 第 i 个积分节点关于原子对 (j, k) 的模糊划分权重
        W_mat = zeros(nintp_atom, natom, natom);
        % Shift the integral point to atom
        rnowx = ip_atom(:, 1) + atom_xyz(iatom, 1);
        rnowy = ip_atom(:, 2) + atom_xyz(iatom, 2);
        rnowz = ip_atom(:, 3) + atom_xyz(iatom, 3);
        % Enumerate each atom pair (j, k)
        for jatom = 1 : natom 
            dxj = rnowx - atom_xyz(jatom, 1);
            dyj = rnowy - atom_xyz(jatom, 2);
            dzj = rnowz - atom_xyz(jatom, 3);
            dij = sqrt(dxj.^2 + dyj.^2 + dzj.^2);
            for katom = 1 : natom
                if katom ~= jatom
                    dxk = rnowx - atom_xyz(katom, 1);
                    dyk = rnowy - atom_xyz(katom, 2);
                    dzk = rnowz - atom_xyz(katom, 3);
                    dik = sqrt(dxk.^2 + dyk.^2 + dzk.^2);
                    smu = (dij - dik) / dist(jatom, katom);
                    
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
        % 积分节点上第 j 个原子的实际权重等于在此处原子 j 与所有其他原子
        % k 之前权重的乘积。归一化后得到以当前原子中心的积分格点属于当前
        % 原子的模糊划分权重。
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