function num = atomic_num_map(key)
	num = 0;
	if     key == 'H'    num = 1;
	elseif key == 'He'   num = 2;
	elseif key == 'C'    num = 6;
	elseif key == 'N'    num = 7;
	elseif key == 'O'    num = 8;
	elseif key == 'F'    num = 9;
	elseif key == 'Al'   num = 13;
	elseif key == 'Cl'   num = 17;
	if (num == 0)
		fprintf('Please add the atomic number of %s to atomic_num_map.m', key);
		error('Atom not in atomic_num_map().');
	end
end
