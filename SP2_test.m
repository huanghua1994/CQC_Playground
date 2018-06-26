function [D, iter] = SP2(F, N, Norb)
% Second-order spectral projection (SP2) algorithm for building density matrix
% D    : Output density matrix
% iter : Number of purification iterations
% F    : Input Fock matrix
% N    : Size of the Fock matrix
% Norb : Number of occupied orbitals, == number of electron / 2

	% Gerschgorin's formula to estimate eigenvalue range
	[Hmax, Hmin] = Gerschgorin_MinMax(F);

	% Generate initial guess
	I = eye(N);
	D = (Hmax .* eye(N) - F) ./ (Hmax - Hmin);
	traceD = trace(D);

	% SP2 iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		D2 = D * D;
		traceD2 = trace(D2);
		tr1 = abs(traceD2 - Norb);
		tr2 = abs(2*traceD - traceD2 - Norb);
		if (tr1 > tr2)
			%D = 2 * D - D2;
			ID2 = I + D2 - 2 * D;
			ID4 = ID2 * ID2;
			D   = I - ID4;
		else
			%D = D2;
			D = D2 * D2;
		end
		traceD = trace(D);

		iter = iter + 1;
		IdemErr = abs(traceD - traceD2);
		if (IdemErr < 1e-11) can_stop = 1; end;
	end
	iter
end