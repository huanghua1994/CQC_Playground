function [V D] = eig_jacobi_onesided_ver1(A)
% [V D] = eig_jacobi_onesided_ver1(A)
%   Parallel Jacobi EVD algorithm (for symmetric matrices).
%   V = matrix of eigenvectors
%   D = vector of eigenvalues (not matrix)
%
%   one_sided_ver1 = one-sided with row access for G and column access for V.
%                    Assumes matrix A is available.

% EC   7/28/2019
% EC   8/01/2019  Replaced eig with symschur2.
% Reference: Golub and van Loan

n = length(A);
if mod(n,2) ~= 0
  error('Input matrix dimension must be even');
end

top = 1:2:n;
bot = 2:2:n;

V = eye(size(A));
G = A; % conceptually, G=V'*A

for sweep = 1:10
  for subsweep = 1:n-1
    for k = 1:n/2
      % choose such that p < q
      p = min(top(k), bot(k));
      q = max(top(k), bot(k));
      % fprintf('zeroing (%d,%d)\n', p, q);

      % calculate block (row access for G; col access for V)
      apq = G(p,:)*V(:,q);
      app = G(p,:)*V(:,p);
      aqq = G(q,:)*V(:,q);

      % calculate J=[c s;-s c] such that J'*Apq*J = diagonal
      [c s] = symschur2([app apq; apq aqq]);

      % update G by applying J' on left (row access)
      tp = G(p,:);
      tq = G(q,:);
      G(p,:) = c*tp - s*tq;
      G(q,:) = s*tp + c*tq;

      % update V by applying J on right (col access)
      tp = V(:,p);
      tq = V(:,q);
      V(:,p) = c*tp - s*tq;
      V(:,q) = s*tp + c*tq;

    end
    [top bot] = music(top, bot);
  end
  fprintf('sweep %2d:  %e\n', sweep, norm(tril(G*V,-1),'fro'));
end
D = zeros(n,1);
for i = 1:n
  D(i) = G(i,:)*V(:,i);
end
