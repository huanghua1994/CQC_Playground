function V = jacobi_block_subsweep(A, rs, re, cs, ce)

n = length(A);
semi_n = n / 2;
V = eye(size(A));
GT = A'; % conceptually, G=V'*A

for p = rs : re
    for q = cs : ce
        if (q <= p), continue; end
    
        % Calculate block (col access for GT; col access for V)
        GTp = GT(:, p);
        GTq = GT(:, q);
        Vp  = V(:, p);
        Vq  = V(:, q);
        apq = dot(GTp, Vq);
        app = dot(GTp, Vp);
        aqq = dot(GTq, Vq);

        % Calculate J=[c s;-s c] such that J'*Apq*J = diagonal
        [c, s] = symschur2([app apq; apq aqq]);

        % Update GT by applying J' on left (col access)
        % Update V by applying J on right (col access)
        GT(:, p) = c * GTp - s * GTq;
        GT(:, q) = s * GTp + c * GTq;
        V(:, p) = c * Vp - s * Vq;
        V(:, q) = s * Vp + c * Vq;
    end
end

end


