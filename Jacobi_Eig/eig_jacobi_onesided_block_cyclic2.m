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
            % Upper triangle blocks
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

            GT(:, blk_p_s:blk_p_e) = GT_blk_p * J_blk(s0:e0, s0:e0) + GT_blk_q * J_blk(s1:e1, s0:e0);
            GT(:, blk_q_s:blk_q_e) = GT_blk_p * J_blk(s0:e0, s1:e1) + GT_blk_q * J_blk(s1:e1, s1:e1);
            V(:, blk_p_s:blk_p_e)  =  V_blk_p * J_blk(s0:e0, s0:e0) +  V_blk_q * J_blk(s1:e1, s0:e0);
            V(:, blk_q_s:blk_q_e)  =  V_blk_p * J_blk(s0:e0, s1:e1) +  V_blk_q * J_blk(s1:e1, s1:e1);
        else
            % Diagonal blocks
            A_blk = GT(:, blk_p_s:blk_p_e)' * V(:, blk_p_s:blk_p_e);
            J_blk = jacobi_block_subsweep(A_blk, 1, blk_p_n, 1, blk_p_n);
            GT(:, blk_p_s:blk_p_e) = GT(:, blk_p_s:blk_p_e) * J_blk;
            V(:, blk_p_s:blk_p_e)  =  V(:, blk_p_s:blk_p_e) * J_blk;
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


