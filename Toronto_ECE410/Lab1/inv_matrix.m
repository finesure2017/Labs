% author: Soon Chee Loong
% This file examines both the accuracy and execution speed of running inv() as well as mldivide (\)
% \ => mldivide => Matrix Left Divide
% / => mrdivide => Matrix Right Divide
n = 500;
% Create orthonormal basis for randomly generated nxn matrix
Q = orth(randn(n, n));
% Create the logspace 
d = logspace(0, -10, n);
% Create a matrix A created by diagonalization => A is diagonalizable
A = Q*diag(d)*Q;
% initialize random vector x
x = randn(n, 1);
% Calculate b
b = A*x;
% Check accuracy of inverse operation to get back x
tic, y = inv(A)*b; toc % Speed for using inv
err = norm(y-x) % Accuracy of resultant x
res = norm(A*y-b) % Accuracy of resultant b
pause(1)
% Check accuracy of using mldivide operation
tic, z = A\b; toc % Speed for using mldivide
err = norm(z-x)
res = norm(A*z - b)
