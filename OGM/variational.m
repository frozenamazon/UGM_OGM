close all
clear
clc

fprintf('Variational inference for Mean field...\n');
variational_meanfield('C:\Users\Lisa\Documents\Matlab\images\map_x.png', [3.0676 1;1 3.0676], ' map');
variational_meanfield('C:\Users\Lisa\Documents\Matlab\images\map5_x.png', [3.0676 1;1 3.0676], ' map 5');
variational_meanfield('C:\Users\Lisa\Documents\Matlab\images\map6_x.png', [3.0676 1;1 3.0676], ' map 6');

pause
fprintf('Variational inference for LBP...\n');
variational_lbp('C:\Users\Lisa\Documents\Matlab\images\map_x.png', [3.0676 1;1 3.0676], ' map');
variational_lbp('C:\Users\Lisa\Documents\Matlab\images\map5_x.png', [3.0676 1;1 3.0676], ' map 5');
variational_lbp('C:\Users\Lisa\Documents\Matlab\images\map6_x.png', [3.0676 1;1 3.0676], ' map 6');
