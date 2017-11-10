n = 1000;
t = 1;
v = [];
dx = .2;
for i = 1:t,
	a = randn(n);
	s = (a + a')/2;
	v = [v; eig(s)];
end
v = v / sqrt(n/2);
[count, x] = hist(v, -2:dx:2);
cla reset
bar(x, count / (t*n*dx), 'y');
hold on;
plot(x, sqrt(4 - x.^2) / (2*pi), 'LineWidth', 2)
axis([-2.5 2.5 -.1 .5]);
