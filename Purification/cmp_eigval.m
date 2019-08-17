function cmp_eigval(F, N, Norb)
	[~, S] = eig(F);
	s = diag(S);
	s = sort(s);
	
	aii = diag(F);
	F = abs(F);
	r = sum(F, 2);
	r = r - abs(aii);
	esti_s = zeros(N, 3);
	esti_s(:, 1) = aii;
	esti_s(:, 2) = aii - r;
	esti_s(:, 3) = aii + r;
	esti_s = sortrows(esti_s);
	
	clf;
	plot(1:N, s, 'r-'), hold on
	plot(Norb, s(Norb), 'r-*'), hold on
	plot(1:N, esti_s(:, 1), 'c--*'), hold on
	plot(1:N, esti_s(:, 2), 'g-'), hold on
	plot(1:N, esti_s(:, 3), 'b-'), hold on
	plot(Norb, esti_s(Norb, 1), 'm-o'), hold on
	grid on
end