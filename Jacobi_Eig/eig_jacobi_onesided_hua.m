function [V, D] = eig_jacobi_onesided_hua(A)
% [V, D] = eig_jacobi_onesided_ver1(A)
%   Parallel Jacobi EVD algorithm (for symmetric matrices).
%   V = matrix of eigenvectors
%   D = vector of eigenvalues (not matrix)
%
%   one_sided_ver1 = one-sided with row access for G and column access for V.
%                    Assumes matrix A is available.

% EC   7/28/2019
% EC   8/01/2019  Replaced eig with symschur2.
% Hua  8/17/2019  Transpose G to access G's columns
% Reference: Golub and van Loan

n = length(A);
if mod(n,2) ~= 0
  error('Input matrix dimension must be even');
end

top = 1:2:n;
bot = 2:2:n;

V = eye(size(A));
%G = A; % conceptually, G=V'*A
GT = A';

for sweep = 1:10
    tic;
    for subsweep = 1:n-1
        % k loop can be directly parallelized 
        for k = 1:n/2
            % Choose such that p < q
            p = min(top(k), bot(k));
            q = max(top(k), bot(k));

            % Calculate block (col access for GT; col access for V)
            GTp = GT(:, p);
            GTq = GT(:, q);
            Vp  = V(:, p);
            Vq  = V(:, q);
            apq = dot(GTp, Vq);
            app = dot(GTp, Vp);
            aqq = dot(GTq, Vq);

            % Calculate J=[c s;-s c] such that J'*Apq*J = diagonal
            % [c s] = symschur2([app apq; apq aqq]);
            if (apq == 0)
                c = 1; s = 0;
            else
                tau = (aqq - app) / (2 * apq);
                if (tau > 0)
                    t =  1 / ( tau + sqrt(1 + tau * tau));
                else
                    t = -1 / (-tau + sqrt(1 + tau * tau));
                end
                c = 1 / sqrt(1 + t * t);
                s = t * c;
            end

            % Update GT by applying J' on left (col access)
            GT(:, p) = c * GTp - s * GTq;
            GT(:, q) = s * GTp + c * GTq;

            % Update V by applying J on right (col access)
            V(:, p) = c * Vp - s * Vq;
            V(:, q) = s * Vp + c * Vq;
        end
        
        %[top, bot] = music(top, bot);
        top_end = top(end);
        top(3 : end) = top(2 : end-1);
        top(2) = bot(1);
        bot(1 : end-1) = bot(2 : end);
        bot(end) = top_end;
    end
    tt = toc;
    fprintf('sweep %2d:  %e, %f\n', sweep, norm(tril(GT' * V,-1),'fro'), tt);
end
D = zeros(n, 1);
for i = 1:n
    D(i) = GT(:, i)' * V(:, i);
end
