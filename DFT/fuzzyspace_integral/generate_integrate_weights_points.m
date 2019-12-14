function [int_points, int_weights] = generate_integrate_weights_points(atom_xyz)
% Get numerical integral points and their fuzzy weights for XC integral
% Ref: [JCP 88, 2547], doi: 10.1063/1.454033, and http://sobereva.com/69
% Input parameter:
%   atom_xyz : Atom coordinates
% Output parameters:
%   int_points  : Numerical integral points (the same for all atoms)
%   int_weights : Fuzzy weights of numerical integral points 
%                 (different for different atoms)

    % The values of nCheby and nLebedev are chosen by the original author
    rm       = 1;    % A parameter, see my_cheb2_becke
    nCheby   = 75;   % Number of radial direction integral points + 1
    nLebedev = 302;  % Number of points on a sphere

    % Generate Lebedev points on a sphere
    Lebedev_points = getLebedevSphere(nLebedev);
    
    % Generate radial direction points & weights
    [rad_r, rad_w] = my_cheb2_becke(nCheby, rm);
    
    % Combine radial direction points with sphere points
    rad_cut_num  = sum(rad_r < 10);
    int_weights0 = zeros(rad_cut_num * nLebedev, 1);
    int_points   = zeros(rad_cut_num * nLebedev, 3);
    orig_points  = [Lebedev_points.x, Lebedev_points.y, Lebedev_points.z];
    for ir = 1 : rad_cut_num
        spos = ir * nLebedev - nLebedev + 1;
        epos = ir * nLebedev;
        int_weights0(spos : epos) = Lebedev_points.w * rad_w(ir);
        int_points(spos : epos, 1 : 3) = orig_points * rad_r(ir);
    end
    nintp = rad_cut_num * nLebedev;  % Total number of integral points
    
    % W_mat(i, j, k): fuzzy weight of integral point i to atom pair (j, k)
    % W_mat(i, j, k): 第 i 个积分节点关于原子对 (j, k) 的模糊划分权重
    natom = size(atom_xyz, 1);
    W_mat = zeros(nintp, natom, natom);
    dist  = squareform(pdist(atom_xyz));
    int_weights = zeros(nintp, natom);
    for iatom = 1 : natom
        % Shift the integral point to atom
        rnowx = int_points(:, 1) + atom_xyz(iatom, 1); 
        rnowy = int_points(:, 2) + atom_xyz(iatom, 2); 
        rnowz = int_points(:, 3) + atom_xyz(iatom, 3); 
        
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
        
        % \prod_{k} W_mat(:, j, k) is the actual weight of integral points
        % belonging to atom k. Normalizing it gives us the fuzzy weight. 
        % 积分节点上第 j 个原子的实际权重等于在此处原子 j 与所有其他原子
        % k 之前权重的乘积。归一化后得到以当前原子中心的积分格点属于当前
        % 原子的模糊划分权重。
        pvec = ones(nintp, natom);
        for i = 1 : natom  
            for j = 1 : natom     
                pvec(:, i) = pvec(:, i) .* W_mat(:, i, j);
            end  
        end
        sum_pvec = sum(pvec, 2);
        int_weights(:, iatom) = int_weights0 .* pvec(:, iatom) ./ sum_pvec;
    end  % end of iatom loop
end  % end function    