function shell = normalize_shell(shell_in)
% Copy of the normalization process in simint/shell/shell.c
    norm_fac = [  
        5.56832799683170785,
        2.78416399841585392,
        4.17624599762378088,
        10.4406149940594522,
        36.5421524792080827,
        164.439686156436372,
        904.418273860400048,
        5878.71878009260031,
        44090.3908506945023,
        374768.322230903270,
        3560299.06119358106,
        37383140.1425326012,
        429906111.639124913,
        5373826395.48906142,
        72546656339.1023291,
        1051926516916.98377,
        16304861012213.2485,
        269030206701518.600,
        4708028617276575.50,
        87098529419616646.7    
    ];
    
    shell  = shell_in;
    nshell = length(shell);
    for i = 1 : nshell
        if (shell(i).alpha(1) == 0)
            if (shell(i).coef(1) ~= 1. || shell(i).nprim ~= 1)
                fprintf('bad unit shell\n');
            end
            continue;
        end
        
        alpha = shell(i).alpha;
        coef  = shell(i).coef;
        nprim = shell(i).nprim;
        am    = shell(i).am;
        m1    = am + 1.5;
        m2    = 0.5 * m1;
        
        N = 0;
        for j = 1 : nprim
            a1 = alpha(j);
            c1 = coef(j);
            for k = 1 : nprim
                a2 = alpha(k);
                c2 = coef(k);
                N = N + (c1 * c2 * (a1 * a2)^m2) / (a1 + a2)^m1;
            end
        end
        shell_norm = 1.0 / sqrt(N * norm_fac(am + 1));
        for j = 1 : nprim
            coef(j) = coef(j) * shell_norm * (alpha(j)^m2);
        end
        shell(i).coef = coef;
    end
end