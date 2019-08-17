function [top bot] = music(top, bot)
% assume top and bot are row vectors
n = size(bot,2);
temp = [top(2:end) fliplr(bot)];
temp = circshift(temp, 1, 2);
top = [1 temp(1:n-1)];
bot = fliplr(temp(n:end));
