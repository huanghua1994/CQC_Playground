function [F, ene_f, ene_d] = DFT(mol_file, max_iter, ene_d_tol)
% DFT calculation
% Input parameters:
%   mol_file  : molecule file
%   max_iter  : maximum SCF iteration
%   ene_d_tol : stop threshold of energy change
% Output parameters:
%   F     : converged Fock matrix
%   ene_f : converged energy, including the nuclear energy
%   ene_d : energy change in each step

    if (nargin < 2), max_iter = 20;     end
    if (nargin < 3), ene_d_tol = 1e-10; end

    [Hcore, S, nbf, nelec, nuc_energy, shells, nshell, shell_bf_num, shell_bf_offsets] = load_mol(mol_file);
    
    nocc   = floor(nelec / 2);
    scrtol2 = 1e-11 * 1e-11;    % Square of shell quartet screening values
    
    % Compute X = S^{-1/2}
    X = inv(chol(S));

    F = zeros(nbf);  % Fock matrix
    D = zeros(nbf);  % Density matrix
    
    Fprime = X' * Hcore * X;
    [C, E] = eig(Fprime);
    [~, index] = sort(diag(E));
    C = X * C;
    C_occ = C(:, index(1 : nocc));
    D = C_occ * C_occ';
    
    max_diis = 10;
    R  = zeros(nbf * nbf, max_diis); % previous residual vectors
    F0 = zeros(nbf * nbf, max_diis); % previous X^T * F * X
    ndiis = 0;
    diis_bmax_id = 1;
    diis_bmax = -19241112;
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
    
    ene_d  = nuc_energy;
    energy = nuc_energy;
    iter   = 0;
    
    tic;
    % Normalize norm_shells for shell_info_to_bf() and eval_Xalpha_XC_with_phi()
    % Don't use it in calculate_eri(), or the shells will be normalized again 
    norm_shells = normalize_shell(shells);
    [natom, atom_xyz, atom_num, bf_coef, bf_alpha, bf_exp, bf_center, bf_nprim] = shell_info_to_bf(nshell, norm_shells);
    [ip, ipw] = generate_int_point_weight(atom_xyz, atom_num);
    phi = eval_bf_at_int_point(ip, nbf, bf_coef, bf_alpha, bf_exp, bf_center, bf_nprim);
    ut = toc;
    fprintf('Precompute bf values at integral points = %.3f (s)\n', ut);
    
    % SCF iterations
    while (ene_d > ene_d_tol)
        tic;
    
        % Construct the Fock matrix
        J = zeros(nbf);
        %K = zeros(nbf);
        
        % M, N, P, Q are shell quartet indices
        for M = 1 : nshell
        for N = 1 : M
        for P = 1 : M
        if (P == M) 
            Qmax = N;
        else
            Qmax = P;
        end
        for Q = 1 : Qmax
            % Shell quartet screening
            MN_scr_val = sq_screen_val(M, N);
            PQ_scr_val = sq_screen_val(P, Q);
            if (MN_scr_val * PQ_scr_val < scrtol2), continue; end

            coef = unique_integral_coef(M, N, P, Q);
            
            % Note: calculate_eri returns row-major tensor, MATLAB use column-major, 
            % so the sequence of indices need to be flipped 
            ERI = calculate_eri(shells(M), shells(N), shells(P), shells(Q));
            ERI = reshape(ERI, [shell_bf_num(Q) shell_bf_num(P) shell_bf_num(N) shell_bf_num(M)]);
            i0  = shell_bf_offsets(M) - 1;
            j0  = shell_bf_offsets(N) - 1;
            k0  = shell_bf_offsets(P) - 1;
            l0  = shell_bf_offsets(Q) - 1;
            
            % i, j, k, l are basis function indices 
            for l = shell_bf_offsets(Q) : shell_bf_offsets(Q + 1) - 1
            for k = shell_bf_offsets(P) : shell_bf_offsets(P + 1) - 1
            for j = shell_bf_offsets(N) : shell_bf_offsets(N + 1) - 1
            for i = shell_bf_offsets(M) : shell_bf_offsets(M + 1) - 1
                I = ERI(l - l0, k - k0, j - j0, i - i0);
                J(i, j) = J(i, j) + coef(1) * D(k, l) * I;
                J(k, l) = J(k, l) + coef(2) * D(i, j) * I;
                %K(i, k) = K(i, k) + coef(3) * D(j, l) * I;
                %K(j, k) = K(j, k) + coef(4) * D(i, l) * I;
                %K(i, l) = K(i, l) + coef(5) * D(j, k) * I;
                %K(j, l) = K(j, l) + coef(6) * D(i, k) * I;
            end
            end
            end
            end
        end
        end
        end
        end

        %J = (J + J') / 2;  % The complete Coulomb matrix
        %K = (K + K') / 2;  % The complete exchange matrix
        %F = Hcore + 2 * J - K;

        % Construct the exchange-correlation matrix
        [XC, Exc] = eval_Xalpha_XC_with_phi(natom, nbf, phi, ipw, D);

        J = (J + J') / 2;
        H = Hcore + 2 * J;
        F = H + XC;
        
        % Calculate energy, note: XC energy is not sum(sum(D.*XC))!!
        prev_energy = energy;
        energy = sum(sum(D .* (Hcore + H))) + Exc + nuc_energy;
        ene_d = abs(energy - prev_energy);
        if (iter == 0)
            ene_d = nuc_energy; 
        else
            ene_del(iter) = ene_d;
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
        C_occ = C(:, index(1 : nocc));
        % D = C_{occ} * C_{occ}^T
        D = C_occ * C_occ';
        
        iter_time = toc;
        
        fprintf('Iteration %2d, energy = %d, energy delta = %d, time = %f\n', iter, energy, ene_d, iter_time);
        iter = iter + 1;
        if (iter > max_iter) 
            break;
        end
    end

    ene_f = energy;
    ene_d = ene_del;
end
