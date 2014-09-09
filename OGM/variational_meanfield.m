function [] = variational_meanfield(imageFile,trainedEdgePot, imageName)

image = imread(imageFile);
grayImage = rgb2gray(image);
[nCols, nRows] = size(grayImage);
figure;
imagesc(image);
title(imageName);

%% Node potential
twoDimensionImage = reshape(grayImage, nCols * nRows, 1);
normalizedImage = double(twoDimensionImage)./255;
nNodes = nCols * nRows;
nodePot = [(1-normalizedImage) normalizedImage];
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

%% Mean field
fprintf('Running Mean Field Inference...\n');
fprintf('Edge potential: ');
trainedEdgePot
[nodeBelMF, edgeBelMF, logZMF] = UGM_Infer_MeanField(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(nodeBelMF(:,2),nCols,nRows));
colormap gray
title(strcat('Mean Field Estimates of Marginals', imageName));