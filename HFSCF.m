function [F, final_energy, energy_delta] = HFSCF(mol_file, max_iter, ene_delta_tol)
% Restrict HF-SCF
% mol_file : molecule file
% max_iter : maximum SCF iteration
% ene_delta_tol : stop threshold of energy change
% F : converged Fock matrix
% final_energy  : converged energy, including the nuclear energy
% energy_delta  : energy change in each step

	if (nargin < 2) max_iter = 20;         end
	if (nargin < 3) ene_delta_tol = 1e-10; end

	[Hcore, S, nbf, nelec, nuc_energy, shells, nshell, shell_bf_num, shell_bf_offsets] = load_mol(mol_file);
	
	n_orb   = nelec / 2;
	scrtol2 = 1e-22;    % Square of shell quartet screening values
	
	% Compute X = S^{-1/2}
	[U, D] = eig(S);
	X = U * inv(sqrt(D)) * U';

	F = zeros(nbf);  % Fock matrix
	D = zeros(nbf);  % Density matrix
	
	load sto3g_alkane20_eri;
	load sto3g_alkane20_den;
	ERI = eri;
	D = D ./ 2;
	
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
		
		ms = shell_bf_offsets(M);
		me = shell_bf_offsets(M + 1) - 1;
		ns = shell_bf_offsets(N);
		ne = shell_bf_offsets(N + 1) - 1;
		
		% Shell quartet (MN|MN)'s ERI tensor
		MNMN_eri = eri(ms:me, ns:ne, ms:me, ns:ne);
		
		% Undone
		% Shell quartet (MN|MN)'s ERI tensor
		% MNMN_eri = calculate_eri(shells(M), shells(N), shells(M), shells(N));
		
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

		% Construct the Fock matrix
		J = zeros(nbf);
		K = zeros(nbf);
		
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
			if (MN_scr_val * PQ_scr_val < scrtol2) continue; end

			coef = unique_integral_coef(M, N, P, Q);
			
			% Undone
			%ERI = calculate_eri(shells(M), shells(N), shells(P), shells(Q));
			%i0  = shell_bf_offsets(M) - 1;
			%j0  = shell_bf_offsets(N) - 1;
			%k0  = shell_bf_offsets(P) - 1;
			%l0  = shell_bf_offsets(Q) - 1;
			
			% i, j, k, l are basis function indices 
			for i = shell_bf_offsets(M) : shell_bf_offsets(M + 1) - 1
			for j = shell_bf_offsets(N) : shell_bf_offsets(N + 1) - 1
			for k = shell_bf_offsets(P) : shell_bf_offsets(P + 1) - 1
			for l = shell_bf_offsets(Q) : shell_bf_offsets(Q + 1) - 1
				I = ERI(i, j, k, l);
				%I = ERI(i - i0, j - j0, k - k0, l - l0);
				J(i, j) = J(i, j) + 2 * coef(1) * D(k, l) * I;
				J(k, l) = J(k, l) + 2 * coef(2) * D(i, j) * I;
				K(i, k) = K(i, k) - coef(3) * D(j, l) * I;
				K(j, k) = K(j, k) - coef(4) * D(i, l) * I;
				K(i, l) = K(i, l) - coef(5) * D(j, k) * I;
				K(j, l) = K(j, l) - coef(6) * D(i, k) * I;
			end
			end
			end
			end
		end
		end
		end
		end
		
		J = (J + J') / 2;  % The complete Coulomb matrix
		K = (K + K') / 2;  % The complete exchange matrix
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

		fprintf('Iteration %2d, energy = %d, energy delta = %d\n', iter, energy, energy_delta);
		iter = iter + 1;
		if (iter > max_iter) 
			break;
		end
	end
end