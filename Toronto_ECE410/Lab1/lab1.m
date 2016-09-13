%----------------------------------------------------------------------------------------
% Create matrix A
%  [1 2 3
%   4 5 6
%   7 8 9]
%----------------------------------------------------------------------------------------
%% Determine row echelon form of A
'LAB 1, 3.I ECHELON FORM'
'Basic 3 x 3 matrix'
A = [1 2 3; 4 5 6; 7 8 9]
'Access 2nd row of A'
% note: Matlab uses 1-indexing instead of 0-indexing
k = 2;
A(k, :)
'Access the 2nd column of A'
A(:, k)
%----------------------------------------------------------------------------------------
'An arbitrary size matrix'
A = [1 0 -2 3 4 0; 0 1 2 4 0 -2; 2 -1 -4 0 -2 8; -3 0 6 -8 -12 2]
'Manually produce reduced row echelon form'
'Fix 1st col'
% '3R - 2 * 1R' 
A(3,:) = A(3,:) - 2*A(1,:);
A(4,:) = A(4,:) + 3*A(1,:)
'Fix 2nd col'
A(3,:) = A(3,:) + A(2,:);
A(3,:) = A(3,:)/2
'Fix 3rd col'
A(1,:) = A(1,:) + 2*A(3,:);
A(2,:) = A(2,:) - 2*A(3,:)
'Fix 4th col'
A(1,:) = A(1,:) - A(4,:);
A(2,:) = A(2,:) - 6*A(4,:);
A(3,:) = A(3,:) + A(4,:)
if A == rref(A)
	'Succeed' 
end 
%----------------------------------------------------------------------------------------
%% Matrix Inverse
% Difference between (Matrix Left Divide) versus Matrix Right Divide
% To solve for x for Ax = B 
% x = A\B; % == (A^-1)B
% To solve for x for xA = B
% x = B/A; % == A(B^-1)
%----------------------------------------------------------------------------------------
% Compute inverse of A using inv(), mldivide(), mrdivide()
'LAB 1, 3.II b) INVERSE'
n = 4;
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
'A is:'
A
'Compute A^-1 using inv()'
inv(A)
'Compute A^-1 using mldivide()'
% Ax = B => (A^-1)(A)x = (A^-1)B  approx. ~ A\B => (A^-1)B(B^-1) = (A\B)(B^-1) approx. ~ (A\B)/B
% (A\b)/b TODO: THIS IS WRONG, FIGURE OUT
pause(1)
'Compute A^-1 using mrdivide()'
% TODO: Figure this out
%----------------------------------------------------------------------------------------
%% Matrix Solution
'LAB 1, 3.II c) SOLUTION OF LINEAR EQUATIONS'
% Find x for Ax = b.
% Ax = B may have no solutions, exactly 1 solution, or inifinitely many. 
A = [1 3 -2 0 2 0; 2 6 -5 -2 4 -3; 0 0 5 10 0 15; 2 6 0 8 4 18]
b = [-4; -11.6; 18; 7.6]
% Use mldivide
'x1 using mlDivide'
x1 = A\b
bx1 = A*x1
% Verify another solution is x2 
x2 = [2; 0; 3; 0; 0; 0.2];
bx2 = A * x2
err = norm(bx2-b)
% Find solution using pseudoinverse pinv()
x3 = pinv(A)*b;
bx3 = A*x3
% Using command norm(), check that ||x3|| < ||x1|| < ||x2||
if (norm(x3) < norm(x1)) && (norm(x1) < norm(x2))
    'Verified that x3 < x1 < x2'
else
    'Unverified'
end 
% What is the main difference between x3 and x1?
diffx1x3 = x1 - x3
diffx1x3norm = norm(x1) - norm(x3)
%----------------------------------------------------------------------------------------
%% Linear Independence, Rank of Matrix
'LAB 1, 3.III a) LINEAR INDEPENDENCE'
% Determine if  linearly independent using rref()
v1 = [1; 3; 1];
v2 = [2; 6; 3];
v3 = [-1; -3; 1];
v4 = [0; 0; 1];
A = transpose(vec2mat([v1; v2; v3; v4], 3))
rref(A)
'It is not linearly independent as contain a row of zeroes in rref'
%%----------------------
'LAB 1, 3.III b) RANK'
% Determine rank using rref(), verify using rank()
A = [1 3 -2 0 2 0; 2 6 -5 -2 4 -3; 0 0 5 10 0 15; 2 6 0 8 4 18]
rref(A)
'It has rank 3 since it has 3 leading ones in rref'
rank(A)
%%----------------------
'LAB 1, 3.III c) COLUMN SPACE'
A = [1 3 -2 0 2 0; 2 6 -5 -2 4 -3; 0 0 5 10 0 15; 2 6 0 8 4 18]
rref(A)
'The nullspace are those at column vectors in which the leading ones are located in rref() which is at position 1, 3, 6'
v1 = [1; 2; 0; 2]
v2 = [-2; -5; 5; 0]
v3 = [0; -3; 15; 18]
%%----------------------
'LAB 1, 3.III d) NULL SPACE'
A = [1 3 -2 0 2 0; 2 6 -5 -2 4 -3; 0 0 5 10 0 15; 2 6 0 8 4 18]
% Determine nullspace using rref(), verify using null(A, 'r')
rref(A)
'Solve from the rref given manually'
null(A,'r') % Correct representation as solved'
null(A) % Orthonormal basis representation
%{
Z = NULL(A) is an orthonormal basis for the null space of A obtained
    from the singular value decomposition.  That is,  A*Z has negligible
    elements, size(Z,2) is the nullity of A, and Z'*Z = I.
Z = NULL(A,'r') is a "rational" basis for the null space obtained
    from the reduced row echelon form.  A*Z is zero, size(Z,2) is an
    estimate for the nullity of A, and, if A is a small matrix with
    integer elements, the elements of R are ratios of small integers.
    The orthonormal basis is preferable numerically, while the rational
    basis may be preferable pedagogically.
%}
%----------------------------------------------------------------------------------------
%% Eigenvalues and eigenvectors
%----------------------------------------------------------------------------------------
'LAB 1, 3.IV a) EIGENVALUES & EIGENVECTORS using eig'
A = [7 2 -3; 4 6 -4 ; 5 2 -1]
[V,D] = eig(A)
%%
A = [7 2 -3; 4 6 -4 ; 5 2 -1]
'LAB 1, 3.IV b) EIGENVALUES & EIGENVECTORS using eig with nobalance'
[V,D] = eig(A, 'nobalance')
% TODO: Do this exercise
% Determine 3 eigenvectors of A whose entries are all integers
