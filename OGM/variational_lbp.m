close all
clc
clear
image = imread('C:\Users\Lisa\Documents\Matlab\images\map6_x.png');
grayImage = rgb2gray(image);
[nCols nRows] = size(grayImage);

%% Node potential
twoDimensionImage = reshape(grayImage, nCols * nRows, 1);
normalizedImage = double(twoDimensionImage)./255;
nodePot = [(1-normalizedImage) normalizedImage];
nNodes = nCols * nRows;
nStates = 2;

%% Adjacent matrix
adj = zeros(nNodes, nNodes);
for i = 1:(nCols)
    for j = 1:(nRows - 1)
        position = i + (j - 1) * nCols;
        adj(position, position + 1) = 1;
        adj(position, position + nCols) = 1;
    end 
end

adj = adj + adj';
edgeStruct = UGM_makeEdgeStruct(adj,nStates);

%% Edge potential
edgePot = repmat([3.0676 1;1 3.0676],[1 1 edgeStruct.nEdges]);

%% Loopy Belief Propagation

fprintf('Running loopy belief propagation for inference...\n');
[nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(nodeBelLBP(:,2),nCols,nRows));
colormap gray
title('Loopy Belief Propagation Estimates of Marginals');
fprintf('(paused)\n');
pause

fprintf('Running loopy belief propagation and computing max of marginals\n');
maxOfMarginalsLBPdecode = UGM_Decode_MaxOfMarginals(nodePot,edgePot,edgeStruct,@UGM_Infer_LBP);

figure;
imagesc(reshape(maxOfMarginalsLBPdecode,nCols,nRows));
colormap gray
title('Max of Loopy Belief Propagation Marginals');
fprintf('(paused)\n');
pause

fprintf('Running loopy belief propagation for decoding...\n');
decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(decodeLBP,nCols,nRows));
colormap gray
title('Loopy Belief Propagation Decoding');
fprintf('(paused)\n');
pause