function [max_ev, min_ev] = Gerschgorin_MinMax(M)
% Gerschgorin's formula to estimate eigenvalue range
	[N, N2] = size(M);
	if (N ~= N2)
		error('Not a square matrix');
	end

	min_ev =  9e99;
	max_ev = -9e99;
	for i = 1 : N
		row_abs_sum = sum(abs(M(i, :)));
		row_abs_sum = row_abs_sum - abs(M(i, i));
		min_ev0 = M(i, i) - row_abs_sum;
		max_ev0 = M(i, i) + row_abs_sum;
		min_ev  = min(min_ev, min_ev0);
		max_ev  = max(max_ev, max_ev0);
	end
end