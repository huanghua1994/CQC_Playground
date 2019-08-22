function [V, D] = eig_jacobi_onesided_block_cyclic2(A, bs)

n = length(A);
A_norm = norm(A, 'fro');
V = eye(size(A));
D = zeros(n, 1);
%G = A; % conceptually, G=V'*A
GT = A';

for sweep = 1:10
    tic;
    for blk_p_s = 1 : bs : n
    for blk_q_s = blk_p_s : bs : n
    
        blk_p_e = min(blk_p_s + bs - 1, n);
        blk_q_e = min(blk_q_s + bs - 1, n);
        
        blk_p_n = blk_p_e - blk_p_s + 1;
        blk_q_n = blk_q_e - blk_q_s + 1;
    
        if (blk_p_s < blk_q_s)
            A_blk = zeros(blk_p_n + blk_q_n);
            s0 = 1; 
            e0 = blk_p_n;
            s1 = blk_p_n+1; 
            e1 = blk_p_n+blk_q_n;
            GT_blk_p = GT(:, blk_p_s:blk_p_e);
            GT_blk_q = GT(:, blk_q_s:blk_q_e);
            V_blk_p  =  V(:, blk_p_s:blk_p_e);
            V_blk_q  =  V(:, blk_q_s:blk_q_e);
            A_blk(s0:e0, s0:e0) = GT_blk_p' * V_blk_p;
            A_blk(s0:e0, s1:e1) = GT_blk_p' * V_blk_q;
            A_blk(s1:e1, s0:e0) = GT_blk_q' * V_blk_p;
            A_blk(s1:e1, s1:e1) = GT_blk_q' * V_blk_q;
            J_blk = jacobi_block_subsweep(A_blk, s0, e0, s1, e1);
            
            %J = eye(n);
            %J(blk_p_s:blk_p_e, blk_p_s:blk_p_e) = J_blk(s0:e0, s0:e0);
            %J(blk_p_s:blk_p_e, blk_q_s:blk_q_e) = J_blk(s0:e0, s1:e1);
            %J(blk_q_s:blk_q_e, blk_p_s:blk_p_e) = J_blk(s1:e1, s0:e0);
            %J(blk_q_s:blk_q_e, blk_q_s:blk_q_e) = J_blk(s1:e1, s1:e1);
            %V = V * J;
            %GT = GT * J;
            GT(:, blk_p_s:blk_p_e) = GT_blk_p * J_blk(s0:e0, s0:e0) + GT_blk_q * J_blk(s1:e1, s0:e0);
            GT(:, blk_q_s:blk_q_e) = GT_blk_p * J_blk(s0:e0, s1:e1) + GT_blk_q * J_blk(s1:e1, s1:e1);
            V(:, blk_p_s:blk_p_e)  =  V_blk_p * J_blk(s0:e0, s0:e0) +  V_blk_q * J_blk(s1:e1, s0:e0);
            V(:, blk_q_s:blk_q_e)  =  V_blk_p * J_blk(s0:e0, s1:e1) +  V_blk_q * J_blk(s1:e1, s1:e1);
        else
            for p = blk_p_s : blk_p_e
            for q = blk_q_s : blk_q_e
                if (q <= p), continue; end  % Skip lower triangle pairs in diagonal blocks
            
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
        end
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


