function[nodePot,edgePot] = trainApprox_pseudo(imageXFile, imageYFile, name)

%% Node potential
% 0 is occupied, 1 is unoccupied

image = imread(imageXFile);
grayImage = rgb2gray(image);
X = (double(grayImage)./255);
figure;
imagesc(X);
colormap gray
title(strcat('Original X', name));

imageY = imread(imageYFile);
grayImageY = rgb2gray(imageY);
Y = int32(idivide(grayImageY,255));
figure;
imagesc(Y);
colormap gray
title(strcat('Original Y', name));

[nRows,nCols] = size(X);
nNodes = nCols * nRows;
nStates = 2;
Y = reshape(Y,[1 1 nNodes]);
X = reshape(X,1,1,nNodes);

%% Make edgeStruct

adj = sparse(nNodes,nNodes);

% Add Down Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],repmat(nRows,[1 nCols]),1:nCols); % No Down edge for last row
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+1)) = 1;

% Add Right Edges
ind = 1:nNodes;
exclude = sub2ind([nRows nCols],1:nRows,repmat(nCols,[1 nRows])); % No right edge for last column
ind = setdiff(ind,exclude);
adj(sub2ind([nNodes nNodes],ind,ind+nRows)) = 1;

% Add Up/Left Edges
adj = adj+adj';
edgeStruct = UGM_makeEdgeStruct(adj,nStates);
nEdges = edgeStruct.nEdges;

%% Make Xnode, Xedge, nodeMap, edgeMap, initialize weights

% Add bias and Standardize Columns
tied = 1;
Xnode = [ones(1,1,nNodes) UGM_standardizeCols(X,tied)];
nNodeFeatures = size(Xnode,2);

% Make nodeMap
nodeMap = zeros(nNodes,nStates,nNodeFeatures,'int32');
for f = 1:nNodeFeatures
    nodeMap(:,1,f) = f;
end

% Make Xedge
sharedFeatures = [1 0];
Xedge = UGM_makeEdgeFeatures(Xnode,edgeStruct.edgeEnds,sharedFeatures);
nEdgeFeatures = size(Xedge,2);

% Make edgeMap
f = max(nodeMap(:));
edgeMap = zeros(nStates,nStates,nEdges,nEdgeFeatures,'int32');
for edgeFeat = 1:nEdgeFeatures
   edgeMap(1,1,:,edgeFeat) = f+edgeFeat;
   edgeMap(2,2,:,edgeFeat) = f+edgeFeat;
end

nParams = max([nodeMap(:);edgeMap(:)]);

%% Train with Pseudo-likelihood

w = zeros(nParams,1);
funObj = @(w)UGM_CRF_PseudoNLL(w,Xnode,Xedge,Y,nodeMap,edgeMap,edgeStruct);
w = minFunc(funObj,w);

%% Evaluate with learned parameters

fprintf('ICM Decoding with estimated parameters...\n');
figure;
[nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
yDecode = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);
imagesc(reshape(yDecode,nRows,nCols));
colormap gray
title(strcat('ICM Decoding with pseudo-likelihood parameters', name));
% fprintf('(paused)\n');
% pause

% %% Now try with non-negative edge features and sub-modular restriction
% 
% sharedFeatures = [2 0];
% Xedge = UGM_makeEdgeFeaturesInvAbsDif(Xnode,edgeStruct.edgeEnds,sharedFeatures);
% nEdgeFeatures = size(Xedge,2);
% 
% % Make different edgeMap
% f = max(nodeMap(:));
% edgeMap = zeros(nStates,nStates,nEdges,nEdgeFeatures,'int32');
% for edgeFeat = 1:nEdgeFeatures
%    edgeMap(1,1,:,edgeFeat) = f+edgeFeat;
%    edgeMap(2,2,:,edgeFeat) = f+edgeFeat;
% end
% 
% nParams = max([nodeMap(:);edgeMap(:)]);
% w = zeros(nParams,1);
% 
% funObj = @(w)UGM_CRF_PseudoNLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct); % Make objective with new Xedge/edgeMap
% UB = [inf;inf;inf;inf]; % No upper bound on parameters
% LB = [-inf;-inf;0;0]; % No lower bound on node parameters, edge parameters must be non-negative 
% w = minConf_TMP(funObj,w,LB,UB);
% 
% fprintf('Graph Cuts Decoding with estimated parameters...\n');
% figure;
% [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
% yDecode = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
% imagesc(reshape(yDecode,nRows,nCols));
% colormap gray
% title('GraphCut Decoding with constrained pseudo-likelihood parameters');
% fprintf('(paused)\n');
% pause
% 
% %% Now try with loopy belief propagation for approximate inference
% 
% w = zeros(nParams,1);
% funObj = @(w)UGM_CRF_NLL(w,Xnode,Xedge,y,nodeMap,edgeMap,edgeStruct,@UGM_Infer_LBP);
% w = minConf_TMP(funObj,w,LB,UB);
% 
% fprintf('Graph Cuts Decoding with estimated parameters...\n');
% figure;
% [nodePot,edgePot] = UGM_CRF_makePotentials(w,Xnode,Xedge,nodeMap,edgeMap,edgeStruct);
% yDecode = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
% imagesc(reshape(yDecode,nRows,nCols));
% colormap gray
% title('GraphCut Decoding with constrained loopy BP parameters');
% fprintf('(paused)\n');

