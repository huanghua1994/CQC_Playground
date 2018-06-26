function [D, iter] = SNS_Purif(F, N, Norb)
% Scaled Newton-Schulz method for purification
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
	
	alpha_hat = 1.6977024852557676;
	x = min(abs(eigval)) / max(abs(eigval)) / 2 + 0.5;

	% Purification iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		alpha = sqrt(3 / (1 - 2 * x + 4 * x * x));
		%alpha = min(alpha, alpha_hat);
		
		D  = (0.5 - 0.75*alpha + 0.25*alpha^3) * I + 1.5*alpha*(1-alpha^2) * D + 3*alpha^3 * D2  - 2*alpha^3 * D3;
		D2 = D * D;
		D3 = D * D2;
		
		x = (0.5 - 0.75*alpha + 0.25*alpha^3)      + 1.5*alpha*(1-alpha^2) * x + 3*alpha^3 * x^2 - 2*alpha^3 * x^3;

		iter = iter + 1;
		err_norm = norm(D - D2, 'fro');
		if (err_norm < 1e-11)   can_stop = 1; end
	end
end
