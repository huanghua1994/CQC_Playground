function [D, iter] = CanonicalPurification(F, N, Ne)
% Canonical Purification
% D    : Output density matrix
% iter : Number of purification iterations
% F    : Input Fock matrix
% N    : Size of the Fock matrix
% Ne   : Number of electron

	% Gerschgorin's formula to estimate eigenvalue range
	Hmin =  9e99;
	Hmax = -9e99;
	for i = 1 : N
		row_abs_sum = sum(abs(F(i, :)));
		row_abs_sum = row_abs_sum - abs(F(i, i));
		Hmin0 = F(i, i) - row_abs_sum;
		Hmax0 = F(i, i) + row_abs_sum;
		Hmin  = min(Hmin, Hmin0);
		Hmax  = max(Hmax, Hmax0);
	end

	% Generate initial guess
	mu_bar = trace(F) / N;
	lambda = min(Ne / (Hmax - mu_bar), (N - Ne) / (mu_bar - Hmin));
	D = (lambda * mu_bar + Ne) / N * eye(N) - lambda / N * F;

	% Purification iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		D2 = D * D;
		D3 = D2 * D;
		c  = trace(D2 - D3) / trace(D - D2);
		if (c <= 0.5)
			D = (1 - 2 * c) * D + (1 + c) * D2 - D3;
			D = D / (1 - c);
		else
			D = (1 + c) * D2 - D3;
			D = D / c;
		end

		iter = iter + 1;
		err_norm = norm(D - D2, 'fro');
		if (err_norm < 1e-11)   can_stop = 1; end
		if ((c < 0) || (c > 1)) can_stop = 1; end
	end
end