function XC = eval_Xalpha_XC_with_phi(natom, nbf, phi, ipw, D)
% Evaluate XC matrix using Xalpha functional and phi
% TODO: find references of formulas used in this function
% Input parameters:
%   nbf : Total number of basis functions
%   phi : Values of basis function at integral points
%   ipw : Integral point weights
%   D   : Density matrix
% Output parameter:
%   XC  : Exchange-correlation matrix

    nintp = size(phi, 1);
    XC = zeros(nbf, nbf);
    
    % rho_g = \sum_{u,v} phi_{g,u} * D_{u,v} * phi_{g,v} is the electron density 
    rho = zeros(nintp, 1);
    for i = 1 : nbf
        rho_j = zeros(nintp, 1);
        for j = 1 : nbf
            rho_j = rho_j + D(i, j) * phi(:, j);
        end
        rho = rho + phi(:, i) .* rho_j;
    end
    
    % Slater exchange energy: E_{XC} = \int -3/4 (3/pi)^(1/3) rho^(4/3) d rho
    % libxc Slater exchange "kernel": exc[rho] = -3/4 (3/pi)^(1/3) rho^(1/3)
    % libxc Xalpha kernel = 3/2 * alpha * exc[rho], alpha = 0.7 includes correlation
    % The last 2^(1/3): we used D = C * C^T instead of D = 2 * C * C^T outside
    exc = -0.7 * (9/8) * ((3/pi)^(1/3)) * rho.^(1/3) * 2^(1/3); 
    
    % The following code is doing: XC_{u,v} = \int exc * phi_u * phi_v d rho,
    % but the formula should be XC_{u,v} = \int fp * phi_u * phi_v d rho, 
    % where f = exc * rho, fp = \frac{\partial f}{\partial rho}. Why?
    exc_w = exc .* ipw;
    for i = 1 : nbf
        phi_i_exc_w = phi(:, i) .* exc_w;
        for j = i : nbf
            XC(i, j) = dot(phi_i_exc_w, phi(:, j));
            XC(j, i) = XC(i, j);
        end
    end
end