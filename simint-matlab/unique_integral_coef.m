function coef = unique_integral_coef(i, j, k, l)
% Handling the coefficient of a shell quartet after symmetric uniqueness check
	flag = zeros(7, 1);
	if (i == j)
		flag(1) = 0;
	else
		flag(1) = 1;
	end

	if (k == l)
		flag(2) = 0;
	else
		flag(2) = 1;
	end
	
	if ((i == k) && (j == l))
		flag(3) = 0;
	else
		flag(3) = 1;
	end
	
	if ((flag(1) == 1) && (flag(2) == 1))
		flag(4) = 1;
	else
		flag(4) = 0;
	end
	
	if ((flag(1) == 1) && (flag(3) == 1))
		flag(5) = 1;
	else
		flag(5) = 0;
	end
	
	if ((flag(2) == 1) && (flag(3) == 1))
		flag(6) = 1;
	else
		flag(6) = 0;
	end
	
	if ((flag(4) == 1) && (flag(3) == 1))
		flag(7) = 1;
	else
		flag(7) = 0;
	end
	
	coef = zeros(6, 1);
	coef(1) = 1       + flag(1) + flag(2) + flag(4); % for J(i, j)
	coef(2) = flag(3) + flag(5) + flag(6) + flag(7); % for J(k, l)
	coef(3) = 1       + flag(3); % for K(i, k)
	coef(4) = flag(1) + flag(5); % for K(j, k)
	coef(5) = flag(2) + flag(6); % for K(i, l)
	coef(6) = flag(4) + flag(7); % for K(j, l)
end