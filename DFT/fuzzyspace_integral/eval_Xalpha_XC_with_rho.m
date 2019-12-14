function XC = eval_Xalpha_XC_with_rho(natom, nbf, rho, ipw, D)
% Evaluate XC matrix using Xalpha functional and rho
% TODO: find references of formulas used in this function
% Input parameters:
%   natom : Total number of atoms
%   nbf   : Total number of basis functions
%   rho   : Values of basis function at integral points
%   ipw   : Integral point weights
%   D     : Density matrix
% Output parameter:
%   XC : Exchange-correlation matrix

    nintp = size(rho, 1);
    XC = zeros(nbf, nbf);
    for k = 1 : natom
        rho_s0 = zeros(nintp, 1);
        for i = 1 : nbf
            rho_s1 = zeros(nintp, 1);
            for j = 1 : nbf
                rho_s1 = rho_s1 + D(i, j) * rho(:, j, k);
            end
            rho_s0 = rho_s0 + rho(:, i, k) .* rho_s1;
        end
        rho_s0 = rho_s0.^(1/3);
        
        for i = 1 : nbf
            for j = i : nbf
                rho_kij  = rho(:, i, k) .* rho(:, j, k) .* rho_s0;
                XC(i, j) = XC(i, j) + dot(rho_kij, ipw(:, k));
                XC(j, i) = XC(i, j);
            end
        end
    end
    
    % Xalpha: v[rho] = -alpha * 3/2 * (3/pi)^{1/3} rho^{1/3}, we use 
    % alpha = 0.7 here (Ref: doi: 10.1016/0301-0104(73)80059-X)
    XC = -0.7 * (3/2) * ((3/pi)^(1/3)) * XC; 
end