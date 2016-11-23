%Setup
clc;

%% run simulation 
m=0.210;
M=0.455;
l=0.303;
g=9.78;

%Initial conditions
x_0 = -0.5;
x_dot_0 = -0.5;
theta_0 = -0.535;
theta_dot_0 = -0.5; 

sim('lab4pt3SFI')
%sim('lab4pt3SFII')
%sim('lab4pt3OFI')
%sim('lab4pt3OFII')