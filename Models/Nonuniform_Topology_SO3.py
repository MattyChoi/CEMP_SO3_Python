# Author: Matthew Choi
# © Regents of the University of Minnesota. All rights reserved
#------------------------------------------------
# generation of the synthetic data
#------------------------------------------------
# Input Parameters: 
# n: the number of the graph nodes
# p: the probability of connecting a pair of vertices. G([n],E) is Erdos-Renyi graph G(n,p).
# p_node_crpt: the probability of corrupting a node
# p_node_edge: the probability of corrupting an edge
# sigma_in: the noise level for inliers
# sigma_out: the noise level for outliers
# crpt_type (optional): choose 'uniform' or 'self-consistent', or 'adv'. The default choice is 'uniform'.

# Output:
# out.AdjMat: n by n adjacency matrix of the generated graph
# out.Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j). edge_num is the number of edges.
# out.RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations
# out.Rij_orig: 3 by 3 by edge_num tensor that stores the ground truth relative rotations
# out.R_orig = R_orig: 3 by 3 by n tensor that stores the ground truth absolute rotations
# out.ErrVec: the true corruption level of each edge
# Reference
# [1] Yunpeng Shi and Gilad Lerman. "Message Passing Least Squares Framework and its Application to Rotation Synchronization" ICML 2020.

import numpy as np
from helpers import *

class SyntheticGraph:
    def __init__(self, AdjMat, Ind, RijMat, Rij_orig, R_orig, ErrVec):
        self.AdjMat = AdjMat
        self.Ind = Ind
        self.RijMat = RijMat
        self.Rij_orig = Rij_orig
        self.R_orig = R_orig
        self.ErrVec = ErrVec

def Nonuniform_Topology_SO3(n, p, p_node_crpt, p_edge_crpt, sigma_in, sigma_out, crpt_type='uniform'):
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

    # create a matrix of indicies
    Ind_full = np.transpose([np.concatenate((Ind_j, Ind_i)), np.concatenate((Ind_i, Ind_j))])

    ### generate rotation matrices
    R_orig = generate_rotations(n)

    # get the number of edges
    m = len(Ind_i)

    Rij_orig = np.zeros((3,3,m))
    IndMat = np.zeros((n,n)).astype(int)
    for k in range(m):
        i, j = Ind_i[k], Ind_j[k] 
        Rij_orig[:,:,k] = R_orig[:,:,i] @ R_orig[:,:,j].T
        IndMat[i,j] = k
        IndMat[j,i] = -k
    
    # get the nodes which the corrupt edges will cluster around
    node_crpt = np.random.permutation(n)
    n_node_crpt = int(n * p_node_crpt)
    node_crpt = node_crpt[:n_node_crpt]

    crptInd = np.zeros(m).astype(bool)

    # only used in self-consistend and adv
    R_crpt = generate_rotations(n)

    ### Add noise to some of the rotations

    # create a matrix that stores the noisy and corrupt relative rotations
    RijMat = Rij_orig.copy()

    for i in node_crpt:
        neighbor_cand = Ind_full[Ind_full[:,0]==i, 1]
        length = len(neighbor_cand)
        neighbor_crpt = np.random.permutation(length)
        n_neighbor = int(p_edge_crpt * length)
        neighbor_crpt = neighbor_crpt[:n_neighbor]
        neighbor_crpt = neighbor_cand[neighbor_crpt]

        for j in neighbor_crpt:
            k = IndMat[i,j]
            crptInd[abs(k)] = True

            R0 = np.random.randn(3,3)
            R0 = project_to_SO3(R0)

            if str.lower(crpt_type) == 'uniform':
                if k >= 0:
                    RijMat[:,:,k] = R0
                else:
                    RijMat[:,:,-k] = R0.T
            else:
                if str.lower(crpt_type) == 'self-consistent':
                    if k >= 0:
                        RijMat[:,:,k] = R_crpt[:,:,i] @ R_crpt[:,:,j].T
                    else:
                        RijMat[:,:,-k] = R_crpt[:,:,j] @ R_crpt[:,:,i].T 
                elif str.lower(crpt_type) == 'adv':
                    if k >= 0:
                        RijMat[:,:,k] = R_crpt[:,:,i] @ R_orig[:,:,j].T
                    else:
                        RijMat[:,:,-k] = R_crpt[:,:,j] @ R_orig[:,:,i].T

    noiseInd = ~crptInd
    # indices of corrupted edges
    RijMat[:,:,noiseInd] += sigma_in * np.random.randn(3, 3, noiseInd.sum())
    RijMat[:,:,crptInd] += sigma_out * np.random.randn(3, 3, crptInd.sum())
    
    # project all rotation matrices back to SO(3)
    project_to_SO3_all(RijMat, m)

    R_err = np.zeros((3,m))
    for j in range(3):
        R_err += Rij_orig[:,j,:] * RijMat[:,j,:]

    # added clipping because of some small floating point errors
    R_err_trace = np.clip(np.sum(R_err, axis=0), -1, 3)
    ErrVec = abs(np.arccos((R_err_trace-1) / 2.0)) / np.pi # computing geodesic distance from the ground truth
    
    return SyntheticGraph(AdjMat, Ind, RijMat, Rij_orig, R_orig, ErrVec)
