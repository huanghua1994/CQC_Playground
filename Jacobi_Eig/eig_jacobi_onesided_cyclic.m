function [V, D] = eig_jacobi_onesided_cyclic(A)

n = length(A);
if mod(n,2) ~= 0
  error('Input matrix dimension must be even');
end

A_norm = norm(A, 'fro');
V = eye(size(A));
D = zeros(n, 1);
%G = A; % conceptually, G=V'*A
GT = A';

for sweep = 1:10
    tic;
    for p = 1 : n-1
        for q = p+1 : n
            % Calculate block (col access for GT; col access for V)
            GTp = GT(:, p);
            GTq = GT(:, q);
            Vp  = V(:, p);
            Vq  = V(:, q);
            apq = dot(GTp, Vq);
            app = dot(GTp, Vp);
            aqq = dot(GTq, Vq);

            % Calculate J=[c s;-s c] such that J'*Apq*J = diagonal
            [c s] = symschur2([app apq; apq aqq]);

            % Update GT by applying J' on left (col access)
            % Update V by applying J on right (col access)
            GT(:, p) = c * GTp - s * GTq;
            GT(:, q) = s * GTp + c * GTq;
            V(:, p) = c * Vp - s * Vq;
            V(:, q) = s * Vp + c * Vq;
        end
    end
    tt = toc;
    for i = 1:n
        D(i) = GT(:, i)' * V(:, i);
    end
    D_norm = norm(D, 2);
    relres_norm = abs(A_norm - D_norm) / A_norm;
    
    fprintf('sweep %2d:  %e, %f\n', sweep, relres_norm, tt);
end


