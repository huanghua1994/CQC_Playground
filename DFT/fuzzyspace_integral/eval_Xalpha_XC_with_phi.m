function [XC, Exc] = eval_Xalpha_XC_with_phi(natom, nbf, phi, ipw, D)
% Evaluate XC matrix using Xalpha functional and phi
% Input parameters:
%   nbf : Total number of basis functions
%   phi : Values of basis function at integral points
%   ipw : Integral point weights
%   D   : Density matrix
% Output parameter:
%   XC  : Exchange-correlation matrix
%   Exc : Exchange-correlation energy

    nintp = size(phi, 1);
    XC = zeros(nbf, nbf);
    
    % rho_g = \sum_{u,v} phi_{g,u} * D_{u,v} * phi_{g,v} is the electron density at g-th grid point
    % Sanity check: \int rho(r) dr = sum(rho .* ipw) ~= total number of electron
    rho = zeros(nintp, 1);
    for i = 1 : nbf
        rho_j = zeros(nintp, 1);
        for j = 1 : nbf
            rho_j = rho_j + D(i, j) * phi(:, j);
        end
        rho = rho + phi(:, i) .* rho_j;
    end
    rho = 2 .* rho;   % We use D = Cocc * Cocc' instead of D = 2 * Cocc * Cocc' outside, need to multiple 2
    
    % Xalpha XC energy: Exc = -alpha * (9/8) * (3/pi)^(1/3) * \int rho(r)^(4/3) dr
    % Xalpha XC potential: vxc(r) = \frac{\delta Exc}{\delta rho} = -alpha * (3/2) * (3*rho(r)/pi)^(1/3)
    % XC_{u,v} = \int phi_u(r) * vxc(r) * phi_v(r) dr
    vxc = -0.7 * (3/2) * ((3/pi)^(1/3)) * rho.^(1/3);
    f   = -0.7 * (9/8) * ((3/pi)^(1/3)) * rho.^(4/3); 
    Exc = sum(f .* ipw);
    vxc_w = vxc .* ipw;
    for i = 1 : nbf
        phi_i_vxc_w = phi(:, i) .* vxc_w;
        for j = i : nbf
            XC(i, j) = dot(phi_i_vxc_w, phi(:, j));
            XC(j, i) = XC(i, j);
        end
    end
end