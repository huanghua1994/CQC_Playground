function [df_tensor, df_nbf] = getDensityFittingTensor(mol_file, df_mol_file)
% Generate 3D density fitting tensor
% Note: some tensor here still use the row-major indexing style, which is
% slow in MATLAB, but just for convenience and for C language reference
% mol_file    : molecule files using normal basis set 
% df_mol_file : molecule files using density fitting basis set
% df_tensor   : 3D density fitting tensor, the last dimension has size df_nbf
% df_nbf      : number of basis functions for density fitting

    shells    = readmol(mol_file);
    df_shells = readmol(df_mol_file);
    df_shells = group_shells_by_AM(df_shells);
    
    nshell    = length(shells);
    df_nshell = length(df_shells);
    
    am  = [shells.am];
    shell_bf_num = (am + 1) .* (am + 2) / 2;  
    nbf = sum(shell_bf_num);
    shell_bf_offset = [1 cumsum(shell_bf_num)+1];
    
    df_am  = [df_shells.am];
    df_shell_bf_num = (df_am + 1) .* (df_am + 2) / 2;
    df_nbf = sum(df_shell_bf_num);
    df_shell_bf_offset = [1 cumsum(df_shell_bf_num)+1];
    
    % Unit shell is a single primitive with orbital exponent zero
    unit_shell.atom_ind = -1;
    unit_shell.atom_sym = 'X';
    unit_shell.am    = 0;
    unit_shell.nprim = 1;
    unit_shell.x     = 0;
    unit_shell.y     = 0;
    unit_shell.z     = 0;
    unit_shell.alpha = 0;
    unit_shell.coef  = 1;
    
    % Calculate 3-center density fitting integrals
    tic;
    pqA = zeros(nbf, nbf, df_nbf);
    for i = 1 : nshell
    for j = i : nshell
        for k = 1 : df_nshell
            eri = calculate_eri(shells(i), shells(j), df_shells(k), unit_shell);
            
            % Convert from row-major to column major
            eri = reshape(eri, df_shell_bf_num(k), shell_bf_num(j), shell_bf_num(i));
            eri = permute(eri, [3 2 1]);
            
            is = shell_bf_offset(i);
            ie = shell_bf_offset(i + 1) - 1;
            js = shell_bf_offset(j);
            je = shell_bf_offset(j + 1) - 1;
            ks = df_shell_bf_offset(k);
            ke = df_shell_bf_offset(k + 1) - 1;
            pqA(is:ie, js:je, ks:ke) = eri;
            
            eri = permute(eri, [2, 1, 3]);
            pqA(js:je, is:ie, ks:ke) = eri;
        end
    end
    end
    three_center_eri_time = toc;
    
    % Calculate the Coulomb metric matrix
    tic;
    Jpq = zeros(df_nbf, df_nbf);
    for i = 1 : df_nshell
    for j = i : df_nshell
        eri = calculate_eri(df_shells(i), unit_shell, df_shells(j), unit_shell);
        
        eri = reshape(eri, df_shell_bf_num(j), df_shell_bf_num(i));
        eri = permute(eri, [2 1]);
        
        is = df_shell_bf_offset(i);
        ie = df_shell_bf_offset(i + 1) - 1;
        js = df_shell_bf_offset(j);
        je = df_shell_bf_offset(j + 1) - 1;
        Jpq(is:ie, js:je) = eri;
        
        eri = permute(eri, [2 1]);
        Jpq(js:je, is:ie) = eri;
    end
    end
    coulomb_metric_mat_time = toc;

    % Form the density fitting tensor, combine with Jpq's inverse square root
    tic;
    pqA0 = permute(pqA, [3 1 2]);
    Jpq_invsqrt = inv(sqrtm(Jpq));
    df_tensor = zeros(nbf, nbf, df_nbf);
    for i = 1 : nbf
    for j = i : nbf
        pqA0_vec = reshape(pqA0(:, i, j), [1, df_nbf]);
        for k = 1 : df_nbf
            %t = 0;
            %for l = 1 : df_nbf
            %    t = t + pqA(i, j, l) * Jpq_invsqrt(l, k);
            %end
            t = pqA0_vec * Jpq_invsqrt(:, k);
            df_tensor(i, j, k) = t;
            df_tensor(j, i, k) = t;
        end
    end
    end
    df_tensor_gen_time = toc;
    
    fprintf('3-center eri, Jpq, DFtensor used time = %f %f %f\n', ...
            three_center_eri_time, coulomb_metric_mat_time, df_tensor_gen_time);
end