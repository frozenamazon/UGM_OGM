#include <math.h>
#include "mex.h"
#include "UGM_common.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int n, s, e, e2, n1, n2, neigh, Vind, Vind2, s1, s2,
            nNodes, nEdges, maxState, dims[3],
            iter, maxIter, 
            *edgeEnds, *nStates, *V, *E, *y;
    
    double *nodePot, *edgePot, *nodeBel, *edgeBel, *logZ,
            z, energy1, energy2, entropy1, entropy2, *prodMsgs, *oldMsgs, *newMsgs, *tmp, *tmp1, *tmp2, *mu,nNbrs;
    
    /* Input */
    
    nodePot = mxGetPr(prhs[0]);
    edgePot = mxGetPr(prhs[1]);
    edgeEnds = (int*)mxGetPr(prhs[2]);
    nStates = (int*)mxGetPr(prhs[3]);
    V = (int*)mxGetPr(prhs[4]);
    E = (int*)mxGetPr(prhs[5]);
    maxIter = ((int*)mxGetPr(prhs[6]))[0];
    mu = mxGetPr(prhs[7]);
    
	if (!mxIsClass(prhs[2],"int32")||!mxIsClass(prhs[3],"int32")||!mxIsClass(prhs[4],"int32")||!mxIsClass(prhs[5],"int32")||!mxIsClass(prhs[6],"int32"))
		mexErrMsgTxt("edgeEnds, nStates, V, E, maxIter must be int32");
	
    /* Compute Sizes */
    
    nNodes = mxGetDimensions(prhs[0])[0];
    maxState = mxGetDimensions(prhs[0])[1];
    nEdges = mxGetDimensions(prhs[2])[0];    
    
    /* Output */
    plhs[0] = mxCreateDoubleMatrix(nNodes,maxState,mxREAL);
    dims[0] = maxState;
    dims[1] = maxState;
    dims[2] = nEdges;
    plhs[1] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    nodeBel = mxGetPr(plhs[0]);
    edgeBel = mxGetPr(plhs[1]);
    logZ = mxGetPr(plhs[2]);
    
    prodMsgs = mxCalloc(maxState*nNodes, sizeof(double));
    oldMsgs = mxCalloc(maxState*nEdges*2, sizeof(double));
    newMsgs = mxCalloc(maxState*nEdges*2, sizeof(double));
    tmp = mxCalloc(maxState, sizeof(double));
    tmp1 = mxCalloc(maxState, sizeof(double));
    tmp2 = mxCalloc(maxState, sizeof(double));
    
    /* Initialize */
    for(e = 0; e < nEdges; e++) {
        n1 = edgeEnds[e]-1;
        n2 = edgeEnds[e+nEdges]-1;
        for(s = 0; s < nStates[n2]; s++)
            newMsgs[s+maxState*e] = 1./nStates[n2];
        for(s = 0; s < nStates[n1]; s++)
            newMsgs[s+maxState*(e+nEdges)] = 1./nStates[n1];
    }
    
    
    
    for(iter = 0; iter < maxIter; iter++) {
        
        for(n=0;n<nNodes;n++) {
            
            /* Update Messages */
            for(Vind = V[n]-1; Vind < V[n+1]-1; Vind++) {
                e = E[Vind]-1;
                n1 = edgeEnds[e]-1;
                n2 = edgeEnds[e+nEdges]-1;
                
                /* First part of message is nodePot*/
                for(s = 0; s < nStates[n]; s++)
                    tmp[s] = nodePot[n + nNodes*s];
                
                /* Multiply by messages from neighbors except j */
                for(Vind2 = V[n]-1; Vind2 < V[n+1]-1; Vind2++) {
                    e2 = E[Vind2]-1;
                    if (e != e2) {
                        if (n == edgeEnds[e2+nEdges]-1) {
                            for(s = 0; s < nStates[n]; s++) {
                                tmp[s] *= pow(newMsgs[s+maxState*e2], mu[e2]);
                            }
                        }
                        else {
                            for(s = 0; s < nStates[n]; s++) {
                                tmp[s] *= pow(newMsgs[s+maxState*(e2+nEdges)], mu[e2]);
                            }
                        }
                    }
                    else {
                        if (n == edgeEnds[e2+nEdges]-1) {
                            for(s = 0; s < nStates[n]; s++) {
                                tmp[s] /= pow(newMsgs[s+maxState*e2], 1.-mu[e2]);
                            }
                        }
                        else {
                            for(s = 0; s < nStates[n]; s++) {
                                tmp[s] /= pow(newMsgs[s+maxState*(e2+nEdges)], 1.-mu[e2]);
                            }
                        }
                    }
                }
                
                /* Now multiply by edge potential to get new message */
                
                if (n == n2) {
                    for(s1 = 0; s1 < nStates[n1]; s1++) {
                        newMsgs[s1+maxState*(e+nEdges)] = 0.0;
                        for(s2 = 0; s2 < nStates[n2]; s2++) {
                            newMsgs[s1+maxState*(e+nEdges)] += tmp[s2]*pow(edgePot[s1+maxState*(s2+maxState*e)], 1./mu[e]);
                        }
                        
                    }
                    
                    /* Normalize */
                    z = 0.0;
                    for(s = 0; s < nStates[n1]; s++)
                        z += newMsgs[s+maxState*(e+nEdges)];
                    for(s = 0; s < nStates[n1]; s++)
                        newMsgs[s+maxState*(e+nEdges)] /= z;
                }
                else {
                    for(s2 = 0; s2 < nStates[n2]; s2++) {
                        newMsgs[s2+maxState*e] = 0.0;
                        for(s1 = 0; s1 < nStates[n1]; s1++) {
                            newMsgs[s2+maxState*e] += tmp[s1]*pow(edgePot[s1+maxState*(s2+maxState*e)], 1./mu[e]);
                        }
                        
                    }
                    
                    /* Normalize */
                    z = 0.0;
                    for(s = 0; s < nStates[n2]; s++)
                        z += newMsgs[s+maxState*e];
                    for(s = 0; s < nStates[n2]; s++)
                        newMsgs[s+maxState*e] /= z;
                }
                
            }
            
            
            
            
        }
        
        /* Print out messages */
        /*
         * printf("\n\nIter = %d\n", iter);
         * for(s=0;s<maxState;s++) {
         * for(e=0;e<nEdges*2;e++) {
         * printf("newMsgs(%d,%d) = %f\n", s, e, newMsgs[s+maxState*e]);
         * }
         * }
         */
        

        /* oldMsgs = newMsgs */
        z = 0;
        for(s=0;s<maxState;s++) {
            for(e=0;e<nEdges*2;e++) {
                z += absDif(newMsgs[s+maxState*e], oldMsgs[s+maxState*e]);
                oldMsgs[s+maxState*e] = newMsgs[s+maxState*e];
            }
        }
        
        /* if sum(abs(newMsgs(:)-oldMsgs(:))) < 1e-4; break; */
        if(z < 1e-4) {
            break;
        }
        
    }
    
    /* ******************* DONE MESSAGE PASSING ********************** */
    
    /*if(iter == maxIter)
     * {
     * printf("LBP reached maxIter of %d iterations\n",maxIter);
     * }
     * printf("Stopped after %d iterations\n",iter); */
    
    /* compute nodeBel */
    for(n = 0; n < nNodes; n++) {
        for(s = 0; s < nStates[n]; s++)
            prodMsgs[s+maxState*n] = nodePot[n+nNodes*s];
        
        for(Vind = V[n]-1; Vind < V[n+1]-1; Vind++) {
            e = E[Vind]-1;
            n1 = edgeEnds[e]-1;
            n2 = edgeEnds[e+nEdges]-1;
            
            if (n == n2) {
                for(s = 0; s < nStates[n]; s++) {
                    prodMsgs[s+maxState*n] *= pow(newMsgs[s+maxState*e], mu[e]);
                }
            }
            else {
                for(s = 0; s < nStates[n]; s++) {
                    prodMsgs[s+maxState*n] *= pow(newMsgs[s+maxState*(e+nEdges)], mu[e]);
                }
            }
        }
        
        z = 0;
        for(s = 0; s < nStates[n]; s++) {
            nodeBel[n + nNodes*s] = prodMsgs[s+maxState*n];
            z = z + nodeBel[n+nNodes*s];
        }
        for(s = 0; s < nStates[n]; s++)
            nodeBel[n + nNodes*s] /= z;
    }
    
    /* Compute edgeBel */
    for(e = 0; e < nEdges; e++) {
        n1 = edgeEnds[e]-1;
        n2 = edgeEnds[e+nEdges]-1;
        
        /* temp1 = nodePot by all messages to n1 except from n2 */
        for(s = 0; s < nStates[n1]; s++)
            tmp1[s] = nodePot[n1 + nNodes*s];
        for(Vind = V[n1]-1; Vind < V[n1+1]-1; Vind++) {
            e2 = E[Vind]-1;
            if (e != e2) {
                if (n1 == edgeEnds[e2+nEdges]-1) {
                    for(s = 0; s < nStates[n1]; s++) {
                        tmp1[s] *= pow(newMsgs[s+maxState*e2],mu[e2]);
                    }
                }
                else {
                    for(s = 0; s < nStates[n1]; s++) {
                        tmp1[s] *= pow(newMsgs[s+maxState*(e2+nEdges)],mu[e2]);
                    }
                }
            }
            else {
                if (n1 == edgeEnds[e2+nEdges]-1) {
                    for(s = 0; s < nStates[n1]; s++) {
                        tmp1[s] /= pow(newMsgs[s+maxState*e2],1.-mu[e2]);
                    }
                }
                else {
                    for(s = 0; s < nStates[n1]; s++) {
                        tmp1[s] /= pow(newMsgs[s+maxState*(e2+nEdges)],1.-mu[e2]);
                    }
                }
            }
        }
        /* temp2 = nodePot by all messages to n2 except from n1 */
        for(s = 0; s < nStates[n2]; s++)
            tmp2[s] = nodePot[n2 + nNodes*s];
        for(Vind = V[n2]-1; Vind < V[n2+1]-1; Vind++) {
            e2 = E[Vind]-1;
            if (e != e2) {
                if (n2 == edgeEnds[e2+nEdges]-1) {
                    for(s = 0; s < nStates[n2]; s++) {
                        tmp2[s] *= pow(newMsgs[s+maxState*e2],mu[e2]);
                    }
                }
                else {
                    for(s = 0; s < nStates[n2]; s++) {
                        tmp2[s] *= pow(newMsgs[s+maxState*(e2+nEdges)],mu[e2]);
                    }
                }
            }
            else {
                if (n2 == edgeEnds[e2+nEdges]-1) {
                    for(s = 0; s < nStates[n2]; s++) {
                        tmp2[s] /= pow(newMsgs[s+maxState*e2],1.-mu[e2]);
                    }
                }
                else {
                    for(s = 0; s < nStates[n2]; s++) {
                        tmp2[s] /= pow(newMsgs[s+maxState*(e2+nEdges)],1.-mu[e2]);
                    }
                }
            }
        }
        
        z = 0;
        for(s1 = 0; s1 < nStates[n1]; s1++) {
            for(s2 = 0; s2 < nStates[n2]; s2++) {
                edgeBel[s1 + maxState*(s2 + maxState*e)] = tmp1[s1]*tmp2[s2]*pow(edgePot[s1+maxState*(s2+maxState*e)],1./mu[e]);
                z += edgeBel[s1+maxState*(s2+maxState*e)];
            }
        }
        for(s1 = 0; s1 < nStates[n1]; s1++) {
            for(s2 = 0; s2 < nStates[n2]; s2++)
                edgeBel[s1+maxState*(s2+maxState*e)] /= z;
        }
    }
    
    /* Compute Bethe Free Energy */
    energy1 = 0;
    energy2 = 0;
    entropy1 = 0;
    entropy2 = 0;
    for(n = 0; n < nNodes; n++) {
        nNbrs = 0;
        for(Vind = V[n]-1; Vind < V[n+1]-1; Vind++)
            nNbrs += mu[E[Vind]-1];
        /*printf("%f\n",nNbrs);*/
        
        for(s = 0; s < nStates[n]; s++) {
            if(nodeBel[n+nNodes*s] > 1e-10)
                entropy1 += (nNbrs-1)*nodeBel[n+nNodes*s]*log(nodeBel[n+nNodes*s]);
            
            energy1 -= nodeBel[n+nNodes*s]*log(nodePot[n+nNodes*s]);
        }
    }
    for(e = 0; e < nEdges; e++) {
        n1 = edgeEnds[e]-1;
        n2 = edgeEnds[e+nEdges]-1;
        
        for(s1 = 0; s1 < nStates[n1];s1++) {
            for(s2 = 0; s2 < nStates[n2]; s2++) {
                if(edgeBel[s1+maxState*(s2+maxState*e)] > 1e-10) {
                    entropy2 -= mu[e]*edgeBel[s1+maxState*(s2+maxState*e)]*log(edgeBel[s1+maxState*(s2+maxState*e)]);
                }
                energy2 -= edgeBel[s1+maxState*(s2+maxState*e)]*log(edgePot[s1+maxState*(s2+maxState*e)]);
            }
        }
    }
    /*printf("%f,%f,%f,%f\n",energy1,energy2,entropy1,entropy2);*/
    logZ[0] = -energy1-energy2+entropy1+entropy2;
    
    
    
    
    /* Free memory */
    mxFree(prodMsgs);
    mxFree(oldMsgs);
    mxFree(newMsgs);
    mxFree(tmp);
    mxFree(tmp1);
    mxFree(tmp2);
}