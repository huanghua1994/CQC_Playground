function [v d] = eig_jacobi_onesided(a)
% [v d] = eig_jacobi_onesided(a)
%   One-sided parallel jacobi evd algorithm (for symmetric matrices); 

% EC   7/28/2019
% Numerically different than classical algorithm.
% This version separates matrix a from gt.

n = length(a);
if mod(n,2) ~= 0
  error('Input matrix dimension must be even');
end

top = 1:2:n;
bot = 2:2:n;

v = eye(size(a));
gt = a; % conceptually, g = v'*a, gt = g' = a'*v
gt = eye(size(a)); % separate a from gt

for sweep = 1:10
  for subsweep = 1:n-1
    for k = 1:n/2
      % choose such that p < q
      p = min(top(k), bot(k));
      q = max(top(k), bot(k));
      % fprintf('zeroing (%d,%d)\n', p, q);

      % calculate block
      % note: apply operator a to a vector
      apq = gt(:,p)'*a*v(:,q); 
      app = gt(:,p)'*a*v(:,p);
      aqq = gt(:,q)'*a*v(:,q);
      aa = [app apq; apq aqq];

      % solve local eigenvalue problem such that vv'*aa*vv = dd
      %[vv dd] = eig(aa);
      [c, s] = symschur2(aa);
      vv = [c s; -s c];

      % update gt by applying vv on right (col access)
      tp = gt(:,p);
      tq = gt(:,q);
      gt(:,p) = vv(1,1)*tp + vv(2,1)*tq;
      gt(:,q) = vv(1,2)*tp + vv(2,2)*tq;

      % update v by applying vv on right (col access)
      vp = v(:,p);
      vq = v(:,q);
      v(:,p) = vv(1,1)*vp + vv(2,1)*vq;
      v(:,q) = vv(1,2)*vp + vv(2,2)*vq;

      % above updates can be done at the same time;
      % could immediately update d(p) = gt(:,p)'*v(:,p) here, and d(q)
    end
    [top bot] = music(top, bot);
  end
  fprintf('sweep %2d:  %e\n', sweep, norm(tril(gt'*a*v,-1),'fro'));
end
d = gt'*a*v; % may not be truly diagonal


function [top bot] = music(top, bot)
% [top bot] = music(top, bot)
%   Generate next set of pairs for elimination.
%   Assumes top and bot are row vectors.
%   Ref: Golub and van Loan.
n = size(bot,2);
temp = [top(2:end) fliplr(bot)];
temp = circshift(temp, 1);
top = [1 temp(1:n-1)];
bot = fliplr(temp(n:end));
