# Author: Matthew Choi
# © Regents of the University of Minnesota. All rights reserved
#------------------------------------------------
# generation of the synthetic data
#------------------------------------------------
# Input Parameters: 
# n: the number of the graph nodes
# p: the probability of connecting a pair of vertices. G([n],E) is Erdos-Renyi graph G(n,p).
# q: the probability of corrupting an edge
# sigma: the noise level (>0)
# crpt_type (optional): choose 'uniform' or 'self-consistent'. The default choice is 'uniform'.

# Output:
# model_out.AdjMat: n by n adjacency matrix of the generated graph
# model_out.Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). edge_num is the number of edges.
# model_out.RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations
# model_out.Rij_orig: 3 by 3 by edge_num tensor that stores the ground truth relative rotations
# model_out.R_orig = R_orig: 3 by 3 by n tensor that stores the ground truth absolute rotations
# model_out.ErrVec: the true corruption level of each edge
# Reference
# [1] Yunpeng Shi and Gilad Lerman. "Message Passing Least Squares Framework and its Application to Rotation Synchronization" ICML 2020.

import numpy as np
import numpy.linalg as LA

class SyntheticGraph:
    def __init__(self, AdjMat, Ind, RijMat, Rij_orig, R_orig, ErrVec):
        self.AdjMat = AdjMat
        self.Ind = Ind
        self.RijMat = RijMat
        self.Rij_orig = Rij_orig
        self.R_orig = R_orig
        self.ErrVec = ErrVec

def Uniform_Topology_SO3(n, p, q, sigma, crpt_type='uniform'):
    ### create an adjacency matrix

    # create random edges
    G = (np.random.random((n,n)) < p).astype(int)   

    # use the connections made in the lower triangular half and set upper half the same
    # to get an undirected graph, also zero out diagonal so no self loops
    G = np.triu(G, 1)
    AdjMat = G + G.T

    ### get the indices of where G has an edge

    # we want i < j which is why we ordered it this way
    Ind_i, Ind_j = np.where(G==1)

    # matrix of size edge_num by 2, each row has the i, j coordinates of each edge
    Ind = np.array([Ind_i, Ind_j]).T

    ### generate rotation matrices
    R_orig = np.zeros((3,3,n))

    for i in range(n):
        Q = np.random.randn(3,3)

        # use svd to get unitary 3x3 matrices
        U, _, V = LA.svd(Q)

        # email yunpeng about this line
        S0 = np.diag([1, 1, LA.det(U @ V)]);  
        R_orig[:,:,i] = U @ S0 @ V
        # R_orig[:,:,i] = LA.det(U @ V) * U @ V

    # get the number of edges
    m = len(Ind_i)

    Rij_orig = np.zeros((3,3,m))
    for k in range(m):
        i, j = Ind_i[k], Ind_j[k]; 
        Rij_orig[:,:,k] = R_orig[:,:,i] @ R_orig[:,:,j].T

    ### Add noise to some of the rotations

    # create a matrix that stores the noisy and corrupt relative rotations
    RijMat = Rij_orig.copy()

    # all edges are noisy or corrupt
    noiseIndLog = np.random.rand(m) >= q
    corrIndLog = ~noiseIndLog

    # get the integer indices
    noiseInd= np.where(noiseIndLog)[0]
    corrInd = np.where(corrIndLog)[0]

    # add the noise
    RijMat[:,:,noiseInd] += sigma * np.random.randn(3, 3, len(noiseInd))

    # project back to SO(3)
    for k in noiseInd:
        # use svd to get unitary 3x3 matrices
        U, _, V = LA.svd(RijMat[:,:,k])

        # email yunpeng about this line
        S0 = np.diag([1, 1, LA.det(U @ V)]);  
        RijMat[:,:,k] = U @ S0 @ V

    if str.lower(crpt_type) == 'uniform':
        for k in corrInd:
            Q = np.random.randn(3,3)

            # use svd to get unitary 3x3 matrices
            U, _, V = LA.svd(Q)

            # email yunpeng about this line
            S0 = np.diag([1, 1, LA.det(U @ V)]);  
            RijMat[:,:,k] = U @ S0 @ V
            
    elif str.lower(crpt_type) == 'self-consistent':
        # create corrupt absolute orientations
        R_corr = np.zeros((3,3,n))

        for i in range(n):
            Q = np.random.randn(3,3)

            # use svd to get unitary 3x3 matrices
            U, _, V = LA.svd(Q)

            # email yunpeng about this line
            S0 = np.diag([1, 1, LA.det(U @ V)]);  
            R_corr[:,:,i] = U @ S0 @ V

        # make corrupt relative rotations for the corrupt edges
        for k in corrInd:
            i, j = Ind_i[k], Ind_j[k]
            Q = R_corr[:,:,i] @ R_corr[:,:,j].T + sigma * np.random.randn(3,3)

            # use svd to get unitary 3x3 matrices
            U, _, V = LA.svd(Q)

            # email yunpeng about this line
            S0 = np.diag([1, 1, LA.det(U @ V)]);  
            RijMat[:,:,k] = U @ S0 @ V

    R_err = np.zeros((3,m))
    for j in range(3):
        R_err += Rij_orig[:,j,:] * RijMat[:,j,:]

    # added clipping because of some small floating point errors
    R_err_trace = np.clip(np.sum(R_err, axis=0), -1, 3)
    ErrVec = abs(np.arccos((R_err_trace-1) / 2.0)) / np.pi; # computing geodesic distance from the ground truth
    
    return SyntheticGraph(AdjMat, Ind, RijMat, Rij_orig, R_orig, ErrVec)

# sg = Uniform_Topology_SO3(200,0.5,0.3,0, crpt_type="uniform")