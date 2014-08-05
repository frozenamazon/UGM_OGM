close all
clc
clear
image = imread('C:\Users\Lisa\Documents\Matlab\images\map0.png');
dimension = size(image);
width = dimension(1);
height = dimension(2);

%% Node potential
twoDimensionImage = reshape(image, width*height, 1);
normalizedImage = double(twoDimensionImage)./255;
nodePot = [normalizedImage (1-normalizedImage)];
nNodes = width * height;
nStates = 2;

%% Adjacent matrix
adj = zeros(nNodes, nNodes);
for i = 1:(width)
    for j = 1:(height - 1)
        position = i + (j - 1) * dimension;
        adj(position, position + 1) = 1;
        adj(position, position+dimension) = 1;
    end 
end

adj = adj + adj';
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
maxState = max(edgeStruct.nStates);

%% Edge potential
edgePot = zeros(maxState,maxState,edgeStruct.nEdges);
for e = 1:edgeStruct.nEdges
   edgePot(:,:,e) = [1 1 ; 1 1]; 
end


% k = [nodeBel(:,1)].*255;
% colorMat = reshape(k, width, height);
% K = mat2gray(colorMat);
% imshow(K)