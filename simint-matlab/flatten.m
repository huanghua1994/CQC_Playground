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
for M=1:m1
  for R=1:m3
    b((M-1)*m2+1:M*m2, (R-1)*m4+1:R*m4) = squeeze(a(M,:,R,:));
  end
end
