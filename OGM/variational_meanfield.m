close all
clc
clear
image = imread('C:\Users\Lisa\Documents\Matlab\images\map6_x.png');
grayImage = rgb2gray(image);
[nCols nRows] = size(grayImage);
figure;
imagesc(image);

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

%% Mean field
fprintf('Running Mean Field Inference...\n');
[nodeBelMF,edgeBelMF,logZMF] = UGM_Infer_MeanField(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(nodeBelMF(:,2),nCols,nRows));
colormap gray
title('Mean Field Estimates of Marginals');