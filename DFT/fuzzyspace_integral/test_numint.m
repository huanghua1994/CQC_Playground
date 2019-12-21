aint = 1.5 * pi^(1.5);  % == \int_{x,y,z=-inf}^{inf} (x^2+y^2+z^2)*exp(-(x^2+y^2+z^2)) dxdydz
for nuc = 1 : 54
    fprintf('Atom %d:\n', nuc);

    % Original scheme
    n_sph = 302;
    n_rad = 75;
    if (nuc <= 10), n_rad = 50; end
    Lebedev_pw = getLebedevSphere(n_sph);
    [rad_r, rad_w] = cheb2_becke(n_rad, 1);
    n_radcut = sum(rad_r < 10); 
    ip_atom   = zeros(n_radcut * n_sph, 3);
    ipw_atom  = zeros(n_radcut * n_sph, 1);
    Lebedev_p = [Lebedev_pw.x, Lebedev_pw.y, Lebedev_pw.z];
    for ir = 1 : n_radcut
        spos = (ir-1) * n_sph + 1;
        epos = ir * n_sph;
        ip_atom(spos : epos, 1 : 3) = Lebedev_p * rad_r(ir);
        ipw_atom(spos : epos) = Lebedev_pw.w * rad_w(ir);
    end
    nintp_atom = n_radcut * n_sph;
    r2 = ip_atom(:, 1).^2 + ip_atom(:, 2).^2 + ip_atom(:, 3).^2;
    f = r2 .* exp(-r2);
    nint = sum(f .* ipw_atom);
    fprintf('    original scheme: %5d points, relerr = %e\n', nintp_atom, (nint - aint) / aint);
    
    % NWChem scheme
    %[n_rad, n_ang] = NWChem_rad_ang_num(nuc);
    n_ang = 302;
    n_rad = 75;
    if (nuc <= 10), n_rad = 50; end
    [rad_r, rad_w] = cheb2_becke(n_rad, 1);
    n_rad = length(rad_r);
    rad_n_ang = NWChem_prune_grid(nuc, n_ang, n_rad, rad_r);
    nintp_atom = sum(rad_n_ang);
    ip_atom  = zeros(nintp_atom, 3);
    ipw_atom = zeros(nintp_atom, 1);
    spos = 1;
    for j = 1 : n_rad
        Lebedev_pw = getLebedevSphere(rad_n_ang(j));
        epos = spos + rad_n_ang(j) - 1;
        ip_atom(spos : epos, 1) = Lebedev_pw.x * rad_r(j);
        ip_atom(spos : epos, 2) = Lebedev_pw.y * rad_r(j);
        ip_atom(spos : epos, 3) = Lebedev_pw.z * rad_r(j);
        ipw_atom(spos : epos) = Lebedev_pw.w * rad_w(j);
        spos = epos + 1;
    end
    r2 = ip_atom(:, 1).^2 + ip_atom(:, 2).^2 + ip_atom(:, 3).^2;
    f = r2 .* exp(-r2);
    nint = sum(f .* ipw_atom);
    fprintf('    NWChem   scheme: %5d points, relerr = %e\n', nintp_atom, (nint - aint) / aint);
end