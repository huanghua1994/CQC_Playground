function [Hcore, S, nbf, nelec, nuc_energy, shells, nshell, shell_bf_num, shell_bf_offsets] = load_mol(fname)
% Load and parse a molecule file (a combination of gbs file and xyz file) and return
% pre-computed core Hamiltonian matrix, overlap matrix and other parameters
% Hcore : core Hamiltonian matrix
% S     : Overlap matrix
% nbf   : total number of basis functions
% nelec : number of electron 
% nuc_energy   : Nuclear energy
% shell_bf_num : number of basis functions in each shell
% shell_bf_offsets : index of the first basis function in each shell

	% Parse molecule file, .mol file already has coordinates in units of Bohr
	% Not sure if the negative charge in the .xyz file can be correctly handled
	shells      = readmol(fname);
	num_atoms   = max([shells.atom_ind]);
	atomic_nums = zeros(num_atoms, 1);
	xcoords     = zeros(num_atoms, 1);
	ycoords     = zeros(num_atoms, 1);
	zcoords     = zeros(num_atoms, 1);
	nshell      = length(shells);
	for i = 1 : nshell
		ind = shells(i).atom_ind;
		atomic_nums(ind) = atomic_num_map(shells(i).atom_sym);
		xcoords(ind)     = shells(i).x;
		ycoords(ind)     = shells(i).y;
		zcoords(ind)     = shells(i).z;
	end
	nelec = sum(atomic_nums);
	
	nuc_energy = 0;
	for i = 1 : num_atoms
		x0 = xcoords(i);
		y0 = ycoords(i);
		z0 = zcoords(i);
		c0 = atomic_nums(i);
		for j = i + 1 : num_atoms
			dx = x0 - xcoords(j);
			dy = y0 - ycoords(j);
			dz = z0 - zcoords(j);
			d  = sqrt(dx * dx + dy * dy + dz * dz);
			nuc_energy = nuc_energy + c0 * atomic_nums(j) / d;
		end
	end
	
	% Calculate core Hamiltonian and overlap matrix
	am = [shells.am];
	shell_bf_num = (am + 1) .* (am + 2) / 2;
	nbf = sum(shell_bf_num);
	shell_bf_offsets = [1 cumsum(shell_bf_num)+1];
	ovl_mat = zeros(nbf, nbf);
	kin_mat = zeros(nbf, nbf);
	pot_mat = zeros(nbf, nbf);
	nshell  = length(shells);
	for i = 1 : nshell
		for j = 1 : nshell
			v = calculate_ovlpi(shells(i), shells(j));
			v = reshape(v, shell_bf_num(j), shell_bf_num(i))';
			ovl_mat(shell_bf_offsets(i):shell_bf_offsets(i+1)-1, shell_bf_offsets(j):shell_bf_offsets(j+1)-1) = v;

			v = calculate_kei(shells(i), shells(j));
			v = reshape(v, shell_bf_num(j), shell_bf_num(i))';
			kin_mat(shell_bf_offsets(i):shell_bf_offsets(i+1)-1, shell_bf_offsets(j):shell_bf_offsets(j+1)-1) = v;

			v = calculate_nai(atomic_nums, xcoords, ycoords, zcoords, shells(i), shells(j));
			v = reshape(v, shell_bf_num(j), shell_bf_num(i))';
			pot_mat(shell_bf_offsets(i):shell_bf_offsets(i+1)-1, shell_bf_offsets(j):shell_bf_offsets(j+1)-1) = v;
		end
	end
	Hcore = kin_mat + pot_mat;
	S = ovl_mat;
end