function [D, iter] = SP2(F, N, Norb)
% Second-order spectral projection (SP2) algorithm for buiding density matrix
% D    : Output density matrix
% iter : Number of purification iterations
% F    : Input Fock matrix
% N    : Size of the Fock matrix
% Norb : Number of occupied orbitals, == number of electron / 2

	% Gerschgorin's formula to estimate eigenvalue range
	[Hmax, Hmin] = Gerschgorin_MinMax(F);

	% Generate initial guess
	D = (Hmax .* eye(N) - F) ./ (Hmax - Hmin);
	traceD = trace(D);

	Ne = Norb * 2;

	% SP2 iterations
	max_iter = 200;
	can_stop = 0;
    iter = 0;
	while ((iter < max_iter) && (can_stop == 0))
		Dtmp = D - D * D;
		traceDtmp = trace(Dtmp);
		if (abs(2*traceD - 2*traceDtmp - Ne) > abs(2*traceD + 2*traceDtmp - Ne))
			D = D + Dtmp;
			traceD = traceD + traceDtmp;
		else
			D = D - Dtmp;
			traceD = traceD - traceDtmp;
		end

		iter = iter + 1;
		IdemErr = abs(traceDtmp);
		if (IdemErr < 1e-12) can_stop = 1; end;
	end
end