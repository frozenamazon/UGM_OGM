close all
clear
clc

fprintf('Training potentials using trainApprox pseudolikelihood...\n');
[nodePot,edgePot] = trainApprox_pseudo('C:\Users\Lisa\Documents\Matlab\images\map_x.png', 'C:\Users\Lisa\Documents\Matlab\images\map_y.png', ' map');

fprintf('Using trained potentials for variational inference...\n');
edgePotential = edgePot(:,:,1);
variational_meanfield('C:\Users\Lisa\Documents\Matlab\images\map_x.png', edgePotential, ' map');
variational_meanfield('C:\Users\Lisa\Documents\Matlab\images\map5_x.png', edgePotential, ' map 5');
variational_meanfield('C:\Users\Lisa\Documents\Matlab\images\map6_x.png', edgePotential, ' map 6');

pause 
fprintf('Variational inference for LBP...\n');
variational_lbp('C:\Users\Lisa\Documents\Matlab\images\map_x.png', edgePotential, ' map');
variational_lbp('C:\Users\Lisa\Documents\Matlab\images\map5_x.png', edgePotential, ' map 5');
variational_lbp('C:\Users\Lisa\Documents\Matlab\images\map6_x.png', edgePotential, ' map 6');