function ret = cheb_filter(k, g, x)
% Calculate the value of a Chebyshev polynomial filtering function
% Input parameters:
%   k : Degree of Chebyshev polynomial
%   g : Center of the amplifying interval
%   x : Input value
% Output parameter:
%   ret : Calculated result
    rho_n = 0;
    rho_d = 0;
    for j = 0 : k
        muhat_kjg = Lanczos_sigma_mu(k, j, g);
        rho_n = rho_n + muhat_kjg * ChebyshevT(j, x);
        rho_d = rho_d + muhat_kjg * ChebyshevT(j, g);
    end
    ret = rho_n ./ rho_d;
end

function ret = ChebyshevT(k, x)
% Calculate the value of the first kind Chebyshev polynomial at a given point
% Input parameters:
%   k : Degree of Chebyshev polynomial
%   x : Input value
% Output parameter:
%   ret : Calculated result
    Tnm2 = 1;
    Tnm1 = x;
    if (k == 0), ret = Tnm2; end
    if (k == 1), ret = Tnm1; end
    if (k > 1)
        for k = 2 : k
            Tn = 2 .* x .* Tnm1 - Tnm2;
            Tnm2 = Tnm1;
            Tnm1 = Tn;
        end
        ret = Tn;
    end
end

function muhat = Lanczos_sigma_mu(k, j, g)
% Calculate the Lanczos sigma-damping expansion coefficient
% Input parameters:
%   k : Total degree of polynomial filter function 
%   j : Current degree of the polynomial
%   g : Center of the amplifying interval
% Output parameter:
%   muhat : Lanczos sigma-damping expansion coefficient
    if (j == 0)
        sigma = 1; 
        mu    = 0.5;
    else
        j_theta_k = j * pi / (k + 1);
        sigma = sin(j_theta_k) / j_theta_k;
        mu    = cos(j * acos(g));
    end
    muhat = mu * sigma;
end