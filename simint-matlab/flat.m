function b = flat(a)
% b = flat(a);
%     a is a 4-index tensor (equi-dimension)
%     b is a square matrix

[m m2 m3 m4] = size(a);
if (m ~= m2) || (m ~= m3) || (m ~= m4)
  error('input must have equal length along all 4 dimensions');
end

b = zeros(m*m,m*m);

% loop over blocks of b
for M=1:m
  for R=1:m
    b((M-1)*m+1:M*m, (R-1)*m+1:R*m) = squeeze(a(M,:,R,:));
  end
end
