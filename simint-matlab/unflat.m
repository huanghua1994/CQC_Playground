function a = unflat(b)
% a = unflat(b)
%     b is a square matrix
%     a is a 4-index tensor (equi-dimension)

m = sqrt(length(b));
a = zeros(m,m,m,m);

% loop over blocks of b
for M=1:m
  for R=1:m
    a(M,:,R,:) = b((M-1)*m+1:M*m, (R-1)*m+1:R*m); 
  end
end

