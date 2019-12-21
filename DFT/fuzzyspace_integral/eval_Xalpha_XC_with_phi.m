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
    phi_D = phi * D';
    rho = sum(phi_D .* phi, 2);
    rho = 2 .* rho;   % We use D = Cocc * Cocc' instead of D = 2 * Cocc * Cocc' outside, need to multiple 2
    
    % Xalpha XC energy: Exc = -alpha * (9/8) * (3/pi)^(1/3) * \int rho(r)^(4/3) dr
    % Xalpha XC potential: vxc(r) = \frac{\delta Exc}{\delta rho} = -alpha * (3/2) * (3*rho(r)/pi)^(1/3)
    % XC_{u,v} = \int phi_u(r) * vxc(r) * phi_v(r) dr
    vxc = -0.7 * (3/2) * ((3/pi)^(1/3)) * rho.^(1/3);
    f   = vxc .* rho .* (3/4); % -0.7 * (9/8) * ((3/pi)^(1/3)) * rho.^(4/3); 
    Exc = sum(f .* ipw);
    vxc_w = vxc .* ipw;
    phi_vxc_w = phi;
    for i = 1 : nbf
        phi_vxc_w(:, i) = phi_vxc_w(:, i) .* vxc_w;
    end
    XC = phi' * phi_vxc_w;
end