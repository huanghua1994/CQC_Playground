function b = flat_fast(a)
% b = flat_fast(a);
%     a is a 4-index tensor (equi-dimension)
%     b is a square matrix
%
% See also: unflat_fast
% These fast versions access a(:,:,P,Q) and b(:,k) which are contiguous.

% For explanation of the pattern, see below.

[m m2 m3 m4] = size(a);
if (m ~= m2) || (m ~= m3) || (m ~= m4)
  error('input must have equal length along all 4 dimensions');
end

b = zeros(m*m,m*m);

% loop over blocks of b
% (MN|PQ)
k = 1;
for P=1:m
  for Q=1:m
    v = a(:,:,P,Q)';
    b(:,k) = v(:);
    k = k + 1;
  end
end

%    >> b=reshape(1:81,9,9);
%    >> b
%    b =
%         1    10    19    28    37    46    55    64    73
%         2    11    20    29    38    47    56    65    74
%         3    12    21    30    39    48    57    66    75
%         4    13    22    31    40    49    58    67    76
%         5    14    23    32    41    50    59    68    77
%         6    15    24    33    42    51    60    69    78
%         7    16    25    34    43    52    61    70    79
%         8    17    26    35    44    53    62    71    80
%         9    18    27    36    45    54    63    72    81
%    >> a=unflat(b);
%    >> a
%    a(:,:,1,1) =
%         1     2     3
%         4     5     6
%         7     8     9
%    a(:,:,2,1) =
%        28    29    30
%        31    32    33
%        34    35    36
%    a(:,:,3,1) =
%        55    56    57
%        58    59    60
%        61    62    63
%    a(:,:,1,2) =
%        10    11    12
%        13    14    15
%        16    17    18
%    a(:,:,2,2) =
%        37    38    39
%        40    41    42
%        43    44    45
%    a(:,:,3,2) =
%        64    65    66
%        67    68    69
%        70    71    72
%    a(:,:,1,3) =
%        19    20    21
%        22    23    24
%        25    26    27
%    a(:,:,2,3) =
%        46    47    48
%        49    50    51
%        52    53    54
%    a(:,:,3,3) =
%        73    74    75
%        76    77    78
%        79    80    81

