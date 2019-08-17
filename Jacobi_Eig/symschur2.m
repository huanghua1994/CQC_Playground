function [c s] = symschur2(a)
% J = symschur2(a)
%  input  a = 2x2 matrix
%  output c,s such that J=[c s; -s c] gives diagonal J'*a*J "close" to a
%  Reference: Golub and van Loan

if a(1,2) == 0
  c = 1;
  s = 0;
  return
end

tau = (a(2,2) - a(1,1))/(2*a(1,2));
if tau > 0
  t =  1/( tau + sqrt(1+tau*tau));
else
  t = -1/(-tau + sqrt(1+tau*tau));
end
c = 1/sqrt(1+t*t);
s = t*c;
