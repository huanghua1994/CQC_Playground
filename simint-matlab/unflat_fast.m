function a = unflat_fast(b)
% a = unflat_fast(b)
%     b is a square matrix
%     a is a 4-index tensor (equi-dimension)

m = sqrt(length(b));
a = zeros(m,m,m,m);

% loop over blocks of b
% (MN|PQ)
k = 1;
for P=1:m
  for Q=1:m
    v = b(:,k);
    a(:,:,P,Q) = reshape(v,m,m)';
    k = k + 1;
  end
end
