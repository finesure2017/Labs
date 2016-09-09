%----------------------------------------------------------------------------------------
% Create matrix A
%  [1 2 3
%   4 5 6
%   7 8 9]
%----------------------------------------------------------------------------------------
%% Determine row echelon form of A
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
% TODO: Do this exercise
A = randn(4, 4);
%----------------------------------------------------------------------------------------
%% Matrix Solution
% Find x for Ax = b.
% Ax = B may have no solutions, exactly 1 solution, or inifinitely many. 
A = [1 3 -2 0 2 0; 2 6 -5 -2 4 -3; 0 0 5 10 0 15; 2 6 0 8 4 18]
b = [-4; -11.6; 18; 17.6]
% Use mldivide
% TODO: Do this exercise
x1 = 0;
% TODO: Do this exercise
% Verify another solution is x2 
x2 = [2; 0; 3; 0; 0; 0.2];
% TODO: Do this exercise
% Find solution using pseudoinverse pinv()
x3 = pinv(A)*b;
% TODO: Do this exercise
% Using command norm(), check that ||x3|| < ||x1|| < ||x2||
% What is the difference between x3 and x1?
% TODO: Do this exercise
%----------------------------------------------------------------------------------------
%% Linear Independence, Rank of Matrix
% Determine if  linearly independent using rref()
% TODO: Do this exercise
v1 = [1; 3; 1];
v2 = [2; 6; 3];
v3 = [-1; -3; 1];
v4 = [0; 0; 1];
% Determine rank using rref() and check using rank()
% TODO: Do this exercise
A = [1 3 -2 0 2 0; 2 6 -5 -2 4 -3; 0 0 5 10 0 15; 2 6 0 8 4 18];

% TODO: Do this exercise
% III(C) 
% III(D) 
%----------------------------------------------------------------------------------------
%% Eigenvalues and eigenvectors
%----------------------------------------------------------------------------------------
