function [i j] = maxmax(a)
% [i j] = maxmax(a);
%   Return indices of max entry in dense matrix a.
%   Caller should use abs if necessary.

% weighted version
% distance from diagonal of i,j is abs(i-j)
%  but nongreedy like this seems slower and decay is slow so that
%   chase nonzeros to the corners
% n = length(a);
% w = zeros(n,n);
% for p = 1:n
% for q = 1:n
%   w(p,q) = 1.5^(abs(p-q));
%   w(p,q) = abs(p-q);
% end
% end
% a = a .* w;

[y, i] = max(a); % row vector of max of each col
[~, j] = max(y);
i = i(j);

