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
	Hmax = max(eigval);  
	Hmin = min(eigval);  
	lambda = min(1 / (Hmax - mu), 1 / (mu - Hmin));
	D  = 0.5 * lambda * (mu * I - F) + 0.5 * I;
	D2 = D * D;
	D3 = D * D2;

	% Purification iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		
		D  = 3 * D2 - 2 * D3;
		D2 = D * D;
		D3 = D * D2;

		iter = iter + 1;
		err_norm = norm(D - D2, 'fro');
		if (err_norm < 1e-11) can_stop = 1; end
	end
end