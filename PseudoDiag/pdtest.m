%rng(19241112);
nbf = 30;
nocc = floor(nbf / 2 - 3);
nvir = nbf - nocc;
F = rand(nbf) + 3 * eye(nbf);
F = F + F';
[C, E] = eig(F);
E = diag(E);
[~, ind] = sort(E);
E = E(ind);
C = C(:, ind);

%% Perturb the Fock matrix a little bit as the next Fock matrix
dF = randn(nbf) * 0.005;
F1 = F + dF + dF';
[C1, E1] = eig(F1);
E1 = diag(E1);
[~, ind1] = sort(E1);
E1 = E1(ind1);
C1 = C1(:, ind1); 
Co1 = C1(:, 1 : nocc);
Cv1 = C1(:, nocc+1 : nbf);
Eo1 = E1(1 : nocc);
Ev1 = E1(nocc+1 : nbf);

%% Use pseudo diagonalization to obtain pseudo eigenvectors
C2 = pseudo_diag(nbf, nocc, C, E, F1);
Co2 = C2(:, 1 : nocc);
Cv2 = C2(:, nocc+1 : nbf);

%u = chol(Co2' * Co2);
%Co2 = Co2 * inv(u);

%% 
D1 = Co1 * Co1';
D2 = Co2 * Co2';