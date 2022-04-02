# Author: Matthew Choi
#------------------------------------------------
# Cycle-Edge Message Passing for Rotation Synchronization
#------------------------------------------------
# Input Parameters: 
# Ind: edge_num by 2 "edge indices matrix". Each row is the index of an edge (i,j) that is sorted as (1,2), (1,3), (1,4),... (2,3), (2,4),.... 
# edge_num is the number of edges.
# RijMat: 3 by 3 by edge_num tensor that stores the given relative rotations corresponding to Ind
# beta_init: initial reweighting parameter beta for CEMP
# beta_max: the maximal reweighting parameter beta for CEMP
# rate: beta is updated by beta = beta*rate until it hits beta_max


# Output:
# SVec: Estimated corruption levels of all edges

# Reference
# [1] Gilad Lerman and Yunpeng Shi. "Robust Group Synchronization via Cycle-Edge Message Passing" arXiv preprint, 2019
# [2] Yunpeng Shi and Gilad Lerman. "Message Passing Least Squares Framework and its Application to Rotation Synchronization" ICML 2020.


import numpy as np
import numpy.linalg as LA
from scipy.sparse import csr_matrix

def CEMP_SO3(Ind, RijMat, beta_init, beta_max, rate):
    ### build the graph

    # get the indices of the edges
    Ind_i, Ind_j = Ind[:,0], Ind[:,1]

    # get the maximal index to build a smaller adjacency matrix in case of very spare matrix
    # does not happen that much in practice
    n = np.max(Ind)+1

    # number of edges
    m = len(Ind_i)

    # create the sparse matrix of edges
    AdjMat = csr_matrix((np.ones(m), (Ind_i, Ind_j)), shape=(n,n))   # Adjacency matrix

    # use symmetry to create a dense matrix
    AdjMat = np.array((AdjMat + AdjMat.T).todense())

    ### start CEMP iterations as initialization   

    # AdjMat @ AdjMat gives all paths of two and element-wise multiplying this with
    # AdjMat identifies all paths of two that is also a path of one (cycle of lenth three)
    CoDeg = (AdjMat @ AdjMat) * AdjMat

    # mark all the pairs of two nodes that have paths of length one but not paths of length two with -1
    CoDeg[(CoDeg == 0) & (AdjMat > 0)] = -1

    # grab the nonzero elements
    CoDeg_upper = np.triu(CoDeg, 1)
    CoDeg_vec = CoDeg_upper.flatten()
    CoDeg_vec = CoDeg_vec[CoDeg_vec != 0]

    # get all the indices in CoDeg_vec that corresponds to 
    CoDeg_pos_ind = np.where(CoDeg_vec > 0)[0]
    CoDeg_vec_pos = CoDeg_vec[CoDeg_pos_ind].astype(int)
    cum_ind = np.insert(np.cumsum(CoDeg_vec_pos), 0, 0)
    m_cycle = cum_ind[-1]
    
    Ind_ij = np.zeros(m_cycle).astype(int)
    Ind_jk = np.zeros(m_cycle).astype(int)
    Ind_ki = np.zeros(m_cycle).astype(int)
    
    RijMat4d = np.zeros((3,3,n,n))
    IndMat = np.zeros((n,n))

    ### construct edge index matrix (for 2d-to-1d index conversion)
    for l in range(m):
        i, j = Ind_i[l], Ind_j[l]
        RijMat4d[:,:,i,j] = RijMat[:,:,l]
        RijMat4d[:,:,j,i] = RijMat[:,:,l].T
        IndMat[i,j] = l
        IndMat[j,i] = l
   
    Rjk0Mat = np.zeros((3,3,m_cycle))
    Rki0Mat = np.zeros((3,3,m_cycle))
    
    m_pos = len(CoDeg_pos_ind)
    for l in range(m_pos):
        IJ = CoDeg_pos_ind[l]
        i, j = Ind_i[IJ], Ind_j[IJ]

        CoInd_ij = np.where(AdjMat[:,i] * AdjMat[:,j])[0]
        Ind_ij[cum_ind[l]:cum_ind[l+1]] = IJ
        Ind_jk[cum_ind[l]:cum_ind[l+1]] = IndMat[j,CoInd_ij]
        Ind_ki[cum_ind[l]:cum_ind[l+1]] = IndMat[CoInd_ij,i]
        Rjk0Mat[:,:,cum_ind[l]:cum_ind[l+1]] = RijMat4d[:,:,j,CoInd_ij]
        Rki0Mat[:,:,cum_ind[l]:cum_ind[l+1]] = RijMat4d[:,:,CoInd_ij,i]
        
    Rij0Mat = RijMat[:,:,Ind_ij.astype(int)]

    print("compute R cycle")

    # R_cycle0 uses np.einsum to make use of parallel computation
    R_cycle0 = np.einsum('ijk,jlk->ilk', Rij0Mat, Rjk0Mat)
    # or
    # R_cycle0 = np.zeros((3,3,m_cycle))
    # for k in range(m_cycle):
    #     if ((k+1) % 1000 == 0):
    #         print('{}/{}'.format(k, m_cycle))
    #     R_cycle0[:,:,k] += Rij0Mat[:,:,k] @ Rjk0Mat[:,:,k]
    
    R_cycle = np.einsum('ijk,jlk->ilk', R_cycle0, Rki0Mat)
    # R_cycle = np.zeros((3,3,m_cycle))
    # for k in range(m_cycle):
        # if ((k+1) % 1000 == 0):
        #     print('{}/{}'.format(k, m_cycle))
        # R_cycle[:,:,k] += R_cycle0[:,:,k] @ Rki0Mat[:,:,k]

    R_trace = np.clip(R_cycle[0,0] + R_cycle[1,1] + R_cycle[2,2], -1, 3)
    S0_long = abs(np.arccos((R_trace-1) / 2.0)) / np.pi # computing geodesic distance from the ground truth

    S0_vec = np.ones(m)
    
    Weight_vec = np.ones(m_cycle)
    S0_weight = S0_long * Weight_vec
    
    for l in range(m_pos):
        IJ = CoDeg_pos_ind[l]
        weighted_sum = np.sum(S0_weight[cum_ind[l]:cum_ind[l+1]])
        weight_total = np.sum(Weight_vec[cum_ind[l]:cum_ind[l+1]])
        S0_vec[IJ] = weighted_sum / weight_total
    
    print('Initialization completed!')
    print('Reweighting Procedure Started ...')
    
    iter = 0
    
    SVec = S0_vec
    beta = beta_init
    while beta <= beta_max:
        Sjk = SVec[Ind_jk]
        Ski = SVec[Ind_ki]
        S_sum = Ski+Sjk
        
        Weight_vec = np.exp(-beta * S_sum)
        S0_weight = S0_long * Weight_vec
    
        for l in range(m_pos):
            IJ = CoDeg_pos_ind[l]
            weighted_sum = np.sum(S0_weight[cum_ind[l]:cum_ind[l+1]])
            weight_total = np.sum(Weight_vec[cum_ind[l]:cum_ind[l+1]])
            SVec[IJ] = weighted_sum / weight_total

        # parameter controling the decay rate of reweighting function
        beta = beta*rate
        
        print('Reweighting Iteration {} Completed!'.format(iter))   
        iter += 1

    return SVec



# import sys
# sys.path.append('../Models')
# from Uniform_Topology_SO3 import *

# model_out = Uniform_Topology_SO3(200, 0.5, 0.3, 0, "uniform")
# Ind = model_out.Ind # matrix of edge indices (m by 2)
# RijMat = model_out.RijMat # given corrupted and noisy relative rotations
# ErrVec = model_out.ErrVec # ground truth corruption levels
# R_orig = model_out.R_orig # ground truth rotations

# CEMP_SO3(Ind, RijMat, 1, 40, 1.2)
# # print(CEMP_SO3(Ind, RijMat, 1, 40, 1.2))