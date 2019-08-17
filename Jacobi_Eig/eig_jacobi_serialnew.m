function [V D counts] = eig_jacobi_serialnew(A)
% [V D] = eig_jacobi_serialnew(A)
%   Parallel Jacobi EVD algorithm (for symmetric matrices).
%   V = matrix of eigenvectors
%   D = vector of eigenvalues (not matrix)

% EC   7/28/2019
% EC   8/01/2019  Replaced eig with symschur2.
% Reference: Golub and van Loan

n = length(A);

counts = zeros(size(A));
V = eye(size(A));

for sweep = 1:10
  for subsweep = 1:n-1
    for k = 1:n/2

      % choose element of max magnitude
      [p q] = maxmax(tril(abs(A),-1));
      if p > q,
        [p q] = deal(q,p);
      end
      % need a real stopping criterion since we are counting
      if abs(A(p,q)) < 1e-9
        D = diag(A);
        return
      end
      counts(p,q) = counts(p,q) + 1;
      % fprintf('zeroing (%d,%d)\n', p, q);

      % calculate J=[c s;-s c] such that J'*Apq*J = diagonal
      [c s] = symschur2(A([p q],[p q]));

      % update A by applying J' on left
      tp = A(p,:);
      tq = A(q,:);
      A(p,:) = c*tp - s*tq;
      A(q,:) = s*tp + c*tq;

      % update A by applying J on right
      tp = A(:,p);
      tq = A(:,q);
      A(:,p) = c*tp - s*tq;
      A(:,q) = s*tp + c*tq;

      % update V by applying J on right
      % This is only needed if eigenvectors are needed
      % and note that V can be efficiently stored implicitly as a sequence of
      % rotations if the number of sweeps is small or if not all the 
      % eigenvectors are needed.
      tp = V(:,p);
      tq = V(:,q);
      V(:,p) = c*tp - s*tq;
      V(:,q) = s*tp + c*tq;

    end
  end
  fprintf('sweep %2d:  %e\n', sweep, norm(tril(A,-1),'fro'));
end
D = diag(A);
