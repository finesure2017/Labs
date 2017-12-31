clear; clc; close all;

A = [0 1 0 0;
    0 0 -4.5138 0;
    0 0 0 1;
    0 0 47.1744 0];

B = [0; 2.1978; 0; -7.2535];
 

P1 = [-1, -2, -3, -4];
P2 = [-1, -1+i, -1-i, -2];

K1 = place(A,B,P1)
K2 = place(A,B,P2)

%open_system('lab4pt2.mdl')
%set_param('lab4pt2.mdl', 'StopTime','60')
%sim('lab4pt2.mdl')


