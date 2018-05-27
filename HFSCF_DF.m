function [F, final_energy, energy_delta] = HFSCF_DF(mol_file, df_mol_file, max_iter, ene_delta_tol)
% Restrict HF-SCF using density fitting
% mol_file : molecule file
% max_iter : maximum SCF iteration
% ene_delta_tol : stop threshold of energy change
% F : converged Fock matrix
% final_energy  : converged energy, including the nuclear energy
% energy_delta  : energy change in each step

    if (nargin < 3) max_iter = 20;         end
    if (nargin < 4) ene_delta_tol = 1e-10; end

    [Hcore, S, nbf, nelec, nuc_energy, shells, nshell, shell_bf_num, shell_bf_offsets] = load_mol(mol_file);
    
    [df_tensor, df_nbf] = getDensityFittingTensor(mol_file, df_mol_file);
    
    n_orb   = nelec / 2;
    scrtol2 = 1e-22;    % Square of shell quartet screening values
    
    % Compute X = S^{-1/2}
    [U, D] = eig(S);
    X = U * inv(sqrt(D)) * U';

    F = zeros(nbf);  % Fock matrix
    D = zeros(nbf);  % Density matrix
    
    max_diis = 10;
    R  = zeros(nbf * nbf, max_diis); % previous residual vectors
    F0 = zeros(nbf * nbf, max_diis); % previous X^T * F * X
    ndiis = 0;
    diis_bmax_id = 1;
    diis_bmax = -9999999999999;
    B = zeros(max_diis + 1) - 1;
    for i = 1 : max_diis + 1
        B(i, i) = 0;
    end
    
    % Compute shell quartet screening values 
    sq_screen_val = zeros(nshell, nshell);
    for M = 1 : nshell
    for N = 1 : nshell
        % Shell quartet (MN|MN)'s ERI tensor
        MNMN_eri = calculate_eri(shells(M), shells(N), shells(M), shells(N));
        
        MNMN_eri = abs(MNMN_eri);
        val = max(MNMN_eri(:));
        
        sq_screen_val(M, N) = val;
    end
    end
    
    energy_delta = nuc_energy;
    energy       = nuc_energy;
    iter = 0;

    % SCF iterations
    while (energy_delta > ene_delta_tol)
        tic;

        % Construct the Fock matrix
        J = zeros(nbf);
        K = zeros(nbf);
        
        DT = D';

        T_J = zeros(df_nbf, 1);
        for p = 1 : df_nbf
            t = 0;
            for k = 1 : nbf
            for l = 1 : nbf
                t = t + DT(l, k) * df_tensor(l, k, p);
            end
            end
            T_J(p) = t;
        end

        T_K = zeros(df_nbf, nbf, nbf);
        for p = 1 : df_nbf
        for k = 1 : nbf
        for j = 1 : nbf
            t = 0;
            for l = 1 : nbf
                t = t + DT(l, k) * df_tensor(l, j, p);
            end
            T_K(k, j, p) = t;
        end
        end
        end

        for i = 1 : nbf
        for j = i : nbf
            t = 0;
            for p = 1 : df_nbf
                t = t + T_J(p) * df_tensor(i, j, p);
            end
            J(i, j) = 2 * t;
            J(j, i) = 2 * t;
        end
        end
        

        for i = 1 : nbf
        for j = i : nbf
            t = 0;
            for p = 1 : df_nbf
            for k = 1 : nbf
                t = t + T_K(k, j, p) * df_tensor(i, k, p);
            end
            end
            K(i, j) = -t;
            K(j, i) = -t;
        end
        end

        F = Hcore + J + K;
        
        % Calculate energy
        prev_energy = energy;
        energy = sum(sum(D .* (Hcore + F))) + nuc_energy;
        energy_delta = abs(energy - prev_energy);
        if iter == 0 
            energy_delta = nuc_energy; 
        else
            ene_del(iter) = energy_delta;
            energys(iter) = energy;
        end
        
        % Commutator DIIS
        if (iter > 1)
            if (ndiis < max_diis)
                ndiis = ndiis + 1;
                diis_idx = ndiis; 
            else
                % replace the F that have the largest 2-norm of its residual vector
                diis_idx = diis_bmax_id;
            end
            
            FDS = F * D * S;
            diis_r = X' * (FDS - FDS') * X;
            
            % B(i, j) = R(:, i) * R(:, j)
            R(:, diis_idx) = reshape(diis_r, nbf * nbf, 1);
            diis_dot = zeros(1, max_diis);
            for i = 1 : ndiis
                diis_dot(i) = R(:, i)' * R(:, diis_idx);
            end
            B(diis_idx, 1 : ndiis) = diis_dot(1 : ndiis);
            B(1 : ndiis, diis_idx) = diis_dot(1 : ndiis);
            
            % Pick an old F that its residual has the largest 2-norm
            for i = 1 : ndiis
                if (B(i, i) > diis_bmax)
                    diis_bmax = B(i, i);
                    diis_bmax_id = i;
                end
            end
            
            % F = X^T * F * X
            F = X' * F * X;
            F0(:, diis_idx) = reshape(F, nbf * nbf, 1);
            
            % Solve the linear system to minimize the linear combination of residuals
            diis_rhs = zeros(ndiis + 1, 1);
            diis_rhs(ndiis + 1) = -1;
            c = B(1 : ndiis + 1, 1 : ndiis + 1) \ diis_rhs;
            
            % Extrapolate
            F = zeros(nbf * nbf, 1);
            for i = 1 : ndiis
                F = F + c(i) * F0(:, i);
            end
            
            F = reshape(F, nbf, nbf);
        else
            F = X' * F * X;
        end
        
        % Diagonalize F' = C' * epsilon * C
        [C, E] = eig(F);
        
        % Form C = X * C', C_{occ} and D
        [~, index] = sort(diag(E));
        C = X * C;
        C_occ = C(:, index(1 : n_orb));
        
        % D = C_{occ} * C_{occ}^T
        D = C_occ * C_occ';

        iter_time = toc;
        
        fprintf('Iteration %2d, energy = %d, energy delta = %d, time = %f\n', iter, energy, energy_delta, iter_time);
        iter = iter + 1;
        if (iter > max_iter) 
            break;
        end
    end

    final_energy = energy;
    energy_delta = ene_del;
end