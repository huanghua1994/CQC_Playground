function C1 = pseudo_diag(nbf, nocc, C, E, F)
% Pseudo-diagonalization of the Fock matrix
% Input parameters:
%   nbf  : Number of basis functions
%   nocc : Number of occupied orbitals
%   C    : Previous occupation matrix C (all previous exact eigenvectors).
%          C(:, 1:nocc) are eigenvectors corresponding to occupied orbitals.
%   E    : All previous exact eigenvalues
%          E(:, 1:nocc) are eigenvalues corresponding to occupied orbitals.
%   F    : New Fock matrix
% Output parameters:
%   C1 : New approximated eigenvectors

    nvir = nbf - nocc;
    Co = C(:, 1 : nocc);
    Cv = C(:, nocc+1 : nbf);
    Eo = E(1 : nocc);
    Ev = E(nocc+1 : nbf);
    
    % Calculate the rotation angles X(i, a) and the threshold
    Sov = Co' * F * Cv;
    X = zeros(nocc, nvir);
    for i = 1 : nocc
        for a = 1 : nvir
            X(i, a) = Sov(i, a) / (Eo(i) - Ev(a));
        end
    end
    X_tol = 0.04 * max(abs(X(:)));
    
    % Perform Givens rotation between occupied and virtual eigenvectors
    Co1 = Co;  Cv1 = Cv;
    for i = 1 : nocc
        for a = 1 : nvir
            if (abs(X(i, a)) < X_tol), continue; end
            s = X(i, a);
            c = sqrt(1 - s * s);
            Co1(:, i) = c .* Co1(:, i) - s .* Cv(:, a);
            Cv1(:, a) = c .* Cv1(:, a) + s .* Co(:, i);
        end
    end
    C1 = [Co1 Cv1];
    
    % Extra step: orthonormalize Co1?
    %{
    u1 = chol(Co1' * Co1);
    Co1 = Co1 * inv(u1);
    C1 = [Co1 Cv1];
    %}
end