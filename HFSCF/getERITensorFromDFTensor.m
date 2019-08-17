function ERI = getERITensorFromDFTensor(M, N, P, Q, shell_bf_num, shell_bf_offsets, df_tensor, df_nbf)
% Get the ERI tensor of a shell quartet from the density fitting tensor
% M, N, P, Q       : shell quartet [MN|PQ]
% shell_bf_num     : number of basis functions in each shell
% shell_bf_offsets : index of the first basis function in each shell
% df_nbf           : number of basis functions for density fitting

    ERI = zeros(shell_bf_num(M), shell_bf_num(N), shell_bf_num(P), shell_bf_num(Q));
    
    for i = 1 : shell_bf_num(M)
    for j = 1 : shell_bf_num(N)
    for k = 1 : shell_bf_num(P)
    for l = 1 : shell_bf_num(Q)
        t = 0;
		ii = i + shell_bf_offsets(M) - 1;
		jj = j + shell_bf_offsets(N) - 1;
		kk = k + shell_bf_offsets(P) - 1;
		ll = l + shell_bf_offsets(Q) - 1;
        for m = 1 : df_nbf
            t = t + df_tensor(ii, jj, m) * df_tensor(ll, kk, m);
        end
        ERI(i, j, k, l) = t;
    end
    end
    end
    end
end