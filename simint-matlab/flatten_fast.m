function b = flatten(a, m1, m2, m3, m4)
% b = flatten(a, m1, m2, m3, m4);
%     a is a 4-index tensor
%     b is a matrix

d = size(a);
if m1*m2*m3*m4 ~= prod(d)
  error('bad input sizes');
end

b = zeros(m1*m2,m3*m4);

% loop over blocks of b
% (MN|PQ)
k = 1;
for P=1:m3
  for Q=1:m4
    v = a(:,:,P,Q)';
    b(:,k) = v(:);
    k = k + 1;
  end
end
