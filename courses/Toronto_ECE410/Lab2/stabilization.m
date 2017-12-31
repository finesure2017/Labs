% Sample Matlab script to facilitate design of stabilizing compensator for an open loop transfer function given by sys_G. 
% The controller C is first order in the script, given by g*(ks+a)/(s+b).\
% The script computes the loop transfer function GC, plots its Nyquist plot, 
% find the transfer function of the closed loop system and determine the closed loop poles.
% Modify the script if you need a second or higher order controller.
k = 1;
a = 1;
b = 1;
num_C=[k a]; % Numerator of C
den_C=[1 b]; % Denominator of C
sys_C=g*tf(num_C,den_C) % calculate system for C
loop=series(sys_G,sys_C); % 
nyquist(loop)
cloop=feedback(loop,1)
[num_cloop,den_cloop]=tfdata(cloop,'v');
roots(den_cloop)