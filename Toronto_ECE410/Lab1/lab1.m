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
A
invA = inv(A)
I = A*inv(A)
% Compute A^-1 using mldivide()
% AA^-1 = I => A^-1 = A\I
invAMlDivide = A\I
pause(1)
% Compute A^-1 using mrdivide()
% A^-1A = I => I/A
invAMrDivide = I/A
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
'LAB 1, 3.IV b) EIGENVALUES & EIGENVECTORS using eig with nobalance'
A = [7 2 -3; 4 6 -4 ; 5 2 -1]
[Vnobalance,D] = eig(A,'nobalance')
% Determine 3 eigenvectors of A whose entries are all integers
eigen3WithIntegerEntries = [1 1 1; 2 1 0; 3 1 1]
%%
'LAB 1, 3.IV c) CHARACTERISTIC POLYNOMIAL'
% Get the coefficients of the polynomial det(sI - A)
% Thus it's s^3 -12s^2 + 44s - 48 = 0
p = poly(A)
% This finds the roots to the polynomial above,
% which results in the eigenvalues
roots(p)

% Note: Can also get p using this
p2 = poly(eig(A))
%%
'LAB 1, 3.IV d) AV - VD'
A = [7 2 -3; 4 6 -4 ; 5 2 -1]
[V,D] = eig(A)
% AV = VD where D is a diagonal matrix consisting of eigenvalues of A 
% Determine norm(AV-VD) , show more significant digits using long() 
format('long')
normOfAvMinusVd = norm(A*V - V*D)
% Determine exact values of eigenvalues and eigenvectors
eigenValues = [2 6 4] % Just by referring to D
% Since (2,1) is 2* of (1,1) and (3,1) is 3* of (1,1)
eigenVectorA = [1; 2; 3]
% Since (1,2), (2,2), and (3,2) are all the same values
eigenVectorB = [1; 1; 1] 
% Since (1,3) and (3,3) are the same values
eigenVectorC = [1; 0; 1]
% Verify AV-VD = 0
verifyZero = A*V - V*D

%% 
'LAB 1, 3.IV e) DIAGONALIZABLE'
A = [7 2 -3; 4 6 -4 ; 5 2 -1]
[V,D] = eig(A)
verifyDiagonal = inv(V) * A * V
%% 
'LAB 1, 3.IV e) REPEATED EIGENVALUES'
A = [1 1; 0 1]
% Show by hand calculation that eigenvalue is 1 with algebraic multiplicity 2 but only has 1 eigenvector => A isn't diagnolizable
% Show that resulting doesn't satisfy AV = VD and can't diagonalize A
[V, D] = eig(A)
% note: Matlab just repeats the eigenvalue twice which will be wrong.
verifyNotDiagonal = inv(V) * A * V
% 
verifyNotZero = A*V - V*D
%% 
'LAB 1, 3.IV f) JORDAN FORM'
A = [0 4 3; 0 20 16; 0 -25 -20]
% When A isn't diagonalizable, can find Jordan Form instead, which is the next best thing to diagonal matrices
JordanForm = jordan(A)
%----------------------------------------------------------------------------------------
%% Ordinary Differential Equations
%----------------------------------------------------------------------------------------
'LAB 1, 4 a) ODE Initialization'
A = [0 1; -4 -2]
B = [0; 4]
C = [1 0]
D = 0
% State Space System 
sys = ss(A, B, C, D)

'LAB 1, 4 b) PLOT STEP RESPONSE'

%{
Step response of dynamic systems.
    [Y,T] = step(SYS) computes the step response Y of the dynamic system SYS. 
    The time vector T is expressed in the time units of SYS and the time 
    step and final time are chosen automatically. For multi-input systems,
    independent step commands are applied to each input channel. If SYS has 
    NY outputs and NU inputs, Y is an array of size [LENGTH(T) NY NU] where 
    Y(:,:,j) contains the step response of the j-th input channel.
 
    For state-space models, 
       [Y,T,X] = step(SYS) 
    also returns the state trajectory X, an array of size [LENGTH(T) NX NU] 
    for a system with NX states and NU inputs.
%}
[Y,T,X]=step(sys);
% The 2 lines from plot represents
plot(T,X)
%%
'LAB 1, 4 c) ZERO INITIAL CONDITION'
A = [0 1; -4 -2]
B = [0; 0]
C = [1 0]
D = 0
sys_init= ss(A, B, C, D)
x0 = [0; 1]
% Determine response due to only initial conditions
%{ 
Initial condition response of state-space models.
    initial(SYS,X0) plots the undriven response of the state-space model SYS 
    (created with SS) with initial condition X0 on the states. This response 
    is characterized by the equations
                         .
      Continuous time:   x = A x ,  y = C x ,  x(0) = x0 
      Discrete time:  x[k+1] = A x[k],  y[k] = C x[k],  x[0] = x0 .
%}
[Y,T,X] = initial(sys_init,x0);
plot(T,X)
%%
'LAB 1, 4 c) NON-ZERO INITIAL CONDITION'
A = [0 1; -4 -2]
B = [0; 0]
C = [1 0]
D = 0
sys_init= ss(A, B, C, D)
x0 = [0; 1]
% Define input to be a sinusoidal wave
t = 0:0.01:20;
u = sin(t);

% Determine 2nd order equation due to initial condition and sinusoid defined above
%{
lsim  Simulate time response of dynamic systems to arbitrary inputs.
    lsim(SYS,U,T) plots the time response of the dynamic system SYS to the
    input signal described by U and T. The time vector T is expressed in the
    time units of SYS and consists of regularly spaced time samples. The 
    matrix U has as many columns as inputs in SYS and its i-th row specifies 
    the input value at time T(i). For example, 
            t = 0:0.01:5;   u = sin(t);   lsim(sys,u,t)  
    simulates the response of a single-input model SYS to the input 
    u(t)=sin(t) during 5 time units.
 
    For discrete-time models, U should be sampled at the same rate as SYS
    (T is then redundant and can be omitted or set to the empty matrix).
    For continuous-time models, choose the sampling period T(2)-T(1) small 
    enough to accurately describe the input U.  lsim issues a warning when
    U is undersampled and hidden oscillations may occur.
          
    lsim(SYS,U,T,X0) specifies the initial state vector X0 at time T(1) 
    (for state-space models only). X0 is set to zero when omitted.
%}
[Y,t,X]=lsim(sys,u,t,x0);
plot(t, X);
