function a = unflatten(b, m1, m2, m3, m4)
% a = unflatten(b, m1, m2, m3, m4);
%     b is a matrix
%     a is a 4-index tensor

% test:
% a0 = reshape(1:120, 5, 4, 3, 2);
% b = flatten(a0, 5, 4, 3, 2);
% a = unflatten(b, 5, 4, 3, 2);
% norm(a0(:)-a(:))

if m1*m2 ~= size(b,1) || m3*m4 ~= size(b,2)
  error('bad input sizes');
end

a = zeros(m1,m2,m3,m4);

% loop over blocks of b
% (MN|PQ)
k = 1;
for P=1:m3
  for Q=1:m4
    v = b(:,k);
    a(:,:,P,Q) = reshape(v,m2,m1)';
    k = k + 1;
  end
end
