function [D, iter] = McWeenyPurification(F, N, Norb)
% McWeeny Purification for density matrix construction
% D    : Output density matrix
% iter : Number of purification iterations
% F    : Input Fock matrix
% N    : Size of the Fock matrix
% Norb : Number of occupied orbitals, == number of electron / 2

	I = eye(N);

	% Solve the eigen problem to find mu and max/min eigenvalues
	% In practice these values should be obtained by other faster methods
	[~, S] = eig(F); 
	eigval = diag(S);
	eigval = sort(eigval);
	mu = (eigval(Norb) + eigval(Norb + 1)) / 2;
	F  = mu * I - F;
	
	[~, S] = eig(F); 
	eigval = diag(S);
	Hmax_abs = max(abs(eigval));
	Hmin_abs = min(abs(eigval));
	D = F ./ Hmax_abs;
	D2 = D * D;

	% Purification iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		
		D = 0.5 * D * (3 * I - D2);
		D2 = D * D;
		realD  = (D + I) ./ 2;
		realD2 = 0.25 * D2 + 0.5 * D + 0.25 * I;

		iter = iter + 1;
		err_norm = norm(realD - realD2, 'fro');
		if (err_norm < 1e-11) can_stop = 1; end
	end

	D = (D + I) ./ 2;
end