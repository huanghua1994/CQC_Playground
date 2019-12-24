function phi = eval_bf_at_int_point(ip, nbf, bf_coef, bf_alpha, bf_exp, bf_center, bf_nprim)
% Evaluate the values of basis function at integral points
% Input parameters:
%   ip         : Integral point coordinates
%   nbf        : Total number of basis functions
%   bf_coef    : coef  terms  of basis functions
%   bf_alpha   : alpha terms  of basis functions
%   bf_exp     : Polynomial exponents terms of basis functions
%   bf_center  : Center of basis functions
%   bf_nprim   : Number of primitive functions in each basis function
% Output parameter:
%   phi(:, i): i-th basis function value at all integral points

    nintp = size(ip, 1);
    phi   = zeros(nintp, nbf);
    for i = 1 : nbf
        bfx  = bf_center(i, 1);
        bfy  = bf_center(i, 2);
        bfz  = bf_center(i, 3);
        bfxe = bf_exp(i, 1);
        bfye = bf_exp(i, 2);
        bfze = bf_exp(i, 3);
        
        poly = (ip(:,1)-bfx).^bfxe .* (ip(:,2)-bfy).^bfye .* (ip(:,3)-bfz).^bfze;
        dx2  = (ip(:,1)-bfx).^2;
        dy2  = (ip(:,2)-bfy).^2;
        dz2  = (ip(:,3)-bfz).^2;
        d2   = dx2 + dy2 + dz2;
        
        for p = 1 : bf_nprim(i)
            phi(:, i) = phi(:, i) + (bf_coef(i, p) .* poly) .* exp(-bf_alpha(i, p) * d2);
        end
    end
end