Q = normrnd(0, 1, [20 20]);
Q = orth(Q);
% Q = eye(20);
for p = 16 : -1 : 2
	e = 10^(-p);
	d = zeros(20, 1);
	d(1) = -e;
	for i = 1 : 17
		d(1 + i) = e^(i / 17);
	end
	d(19) = 1;
	d(20) = 1;
	d = sort(d);
	F = Q' * diag(d) * Q;

	[D1, it1] = McWeenyPurification(F, 20, 1);
	[D2, it2] = CanonicalPurification(F, 20, 1);
	[D3, it3] = SSNS_Purif(F, 20, 1);
	[D4, it4] = SNS_Purif(F, 20, 1);
	fprintf('%d %d %d %d\n', it1, it2, it3, it4);
end