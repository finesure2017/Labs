%%For finding eigenvalues for single value of k
clc;
clear;

x_star = [0; 0; 0; 0];
u_star = 0;

   
syms x1 x2 x3 x4;
x = [x1; x2; x3; x4];

syms u;

m=0.210;
M=0.455;
l=0.303;
g=9.78;

%syms m M l g;

syms f1(x1, x2, x3, x4, u);
syms f2(x1, x2, x3, x4, u);
syms f3(x1, x2, x3, x4, u);
syms f4(x1, x2, x3, x4, u);

f1(x1,x2,x3,x4,u) = x2;
f2(x1,x2,x3,x4,u) = (u + m*l*(x4)^2*sin(x3)-m*g*cos(x3)*sin(x3))/(M+m*(sin(x3))^2);
f3(x1,x2,x3,x4,u) = x4;
f4(x1,x2,x3,x4,u) = ((M+m)*g*sin(x3)-u*cos(x3)-m*l*(x4)^2*sin(x3)*cos(x3))/(l*(M+m*(sin(x3)^2)));

f = [f1; f2; f3; f4];

D = diff(f,x1);
E = diff(f,x2);
F = diff(f,x3);
G = diff(f,x4);
H = diff(f,u);

A_deriv = [D, E, F, G]
B_deriv = H

A = double(A_deriv(x_star(1), x_star(2), x_star(3), x_star(4), u_star))
B = double(B_deriv(x_star(1), x_star(2), x_star(3), x_star(4),u_star))

%A = A_deriv(x_star(1), x_star(2), x_star(3), x_star(4), u_star)
%B = B_deriv(x_star(1), x_star(2), x_star(3), x_star(4),u_star)



eig(A)


%% Finding the transfer function

%Remember to give actual values for M, m, l,g; convert A and B to double

C = [1 0 0 0];
D = 0;
[Num,Den] = ss2tf(A,B,C,D) %The coefficients
sys = tf(Num,Den) %The transfer function

poles = roots(Den)
zeros = roots(Num)



