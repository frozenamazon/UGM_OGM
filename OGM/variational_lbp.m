function [] = variational_lbp(imageFile,trainedEdgePot, imageName)

image = imread(imageFile);
grayImage = rgb2gray(image);
[nCols, nRows] = size(grayImage);

%% Node potential
twoDimensionImage = reshape(grayImage, nCols * nRows, 1);
normalizedImage = double(twoDimensionImage)./255;
nNodes = nCols * nRows;
nodePot = [(1-normalizedImage) ones(nNodes,1)];
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
edgePot = repmat(trainedEdgePot,[1 1 edgeStruct.nEdges]);

%% Loopy Belief Propagation

fprintf('Running loopy belief propagation for inference...\n');
fprintf('Edge potential: ');
trainedEdgePot
[nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(nodeBelLBP(:,2),nCols,nRows));
colormap gray
title(strcat('Loopy Belief Propagation Estimates of Marginals', imageName));
fprintf('(paused)\n');