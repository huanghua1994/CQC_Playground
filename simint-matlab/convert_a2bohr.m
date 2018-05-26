function shells = convert_a2bohr(shells)
% shells = convert_a2bohr(shells)
%   Multiply coordinates by 1.0/0.52917720859

a2bohr = 1.0/0.52917720859;

for i = 1:length(shells)
  shells(i).x = shells(i).x * a2bohr;
  shells(i).y = shells(i).y * a2bohr;
  shells(i).z = shells(i).z * a2bohr;
end
