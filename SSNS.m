function [D, iter] = SSNS(F, N, Norb)
% Stable, Scaled Newton-Schulz method 
% D    : Output density matrix
% iter : Number of purification iterations
% F    : Input Fock matrix
% N    : Size of the Fock matrix
% Norb : Number of occupied orbitals, == number of electron / 2

	I = eye(N);

	% Solve the eigen problem to find mu and max/min eigenvalues
	% In practice these values should be obtained by other faster methods
	[U, S] = eig(F); 
	eigval = diag(S);
	eigval = sort(eigval);
	mu = (eigval(Norb) + eigval(Norb + 1)) / 2;
	F  = mu * I - F;
	
	[U, S] = eig(F); 
	eigval = diag(S);
	Hmax_abs = max(abs(eigval));
	Hmin_abs = min(abs(eigval));
	D = F ./ Hmax_abs;
	D2 = D * D;
	min_ev = Hmin_abs / Hmax_abs;
	alpha_hat = 1.69770248525577;

	% Purification iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		alpha = sqrt(3 / (1 + min_ev + min_ev * min_ev));
		alpha = min(alpha, alpha_hat);
		
		D = 0.5 * alpha * D * (3 * I - alpha * alpha * D2);
		D2 = D * D;
		min_ev = 0.5 * alpha * min_ev * (3 - alpha * alpha * min_ev * min_ev);

		realD  = (D + I) ./ 2;
		realD2 = 0.25 * D2 + 0.5 * D + 0.25 * I;

		iter = iter + 1;
		err_norm = norm(realD - realD2, 'fro');
		if (err_norm < 1e-11)   can_stop = 1; end
	end

	D = (D + I) ./ 2;

	fprintf('iter = %d, err_norm = %d\n', iter, err_norm);
end