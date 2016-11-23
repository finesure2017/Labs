A = [-0.4 0 -0.01; 1 0 0; -1.4 9.8 -0.02];
B = [6.3; 0; 9.8];
C = [0 0 1];
D = 0;
systemG = ss(A,B,C,D); % Generate the system
transferG = tf(systemG); 
%{
      9.8 s^2 - 4.9 s + 61.74
  --------------------------------
  s^3 + 0.42 s^2 - 0.006 s + 0.098
%}

[numeratorG, denominatorG] = ss2tf(A, B, C, D) % 9.8s^2 -4.9s + 61.74

zerosG = roots(numeratorG) % 9.8 s^2 - 4.9 s + 61.74
% Both are open loop unstable due to presence of RE(s) >= 0
polesG = roots(denominatorG) % s^3 + 0.42 s^2 - 0.006 s + 0.098
eigenValueOfA = eig(A); 

% 4.1.1 a) 
% C = [0 0 1] implies y = x3 
% To show all x1, x2, x3, need set C = [1 1 1]

% 4.1.1 b) 

