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
from scipy.sparse import csr_matrix

def CEMP_SO3(Ind, RijMat, beta_init, beta_max, rate):
    ### build the graph

    # get the indices of the edges
    Ind_i, Ind_j = Ind[:,0], Ind[:,1]

    # get the maximal index to build a smaller adjacency matrix in case of very spare matrix
    # does not happen that much in practice
    num_nodes = np.max(Ind)+1

    # number of edges
    num_edges = len(Ind_i)

    # create the sparse matrix of edges
    AdjMat = csr_matrix((np.ones(num_edges), (Ind_i, Ind_j)), shape=(num_nodes,num_nodes))   # Adjacency matrix

    # use symmetry to create a dense matrix
    AdjMat = np.array((AdjMat + AdjMat.T).todense())

    ### start CEMP iterations as initialization   

    # AdjMat @ AdjMat gives all paths of two and element-wise multiplying this with
    # AdjMat identifies all paths of two that is also a path of one (cycle of lenth three)
    cycles = (AdjMat @ AdjMat) * AdjMat

    # mark all the pairs of two nodes that have paths of length one but not paths of length two with -1
    cycles[(AdjMat > 0) & (cycles == 0)] = -1

    # the elements of cycles:
    # --has path of length 1 but not length 2: -1
    # --has path of length 1 and path of length 2: # of such triangles
    # --has no path of length 1 nor 2: 0

    # grab the nonzero elements
    cycles_upper = np.triu(cycles, 1)
    cycles_vec = cycles_upper.flatten()
    cycles_vec = cycles_vec[cycles_vec != 0]

    # get all the indices in cycles_vec that corresponds to traingle
    cycles_pos_ind = np.where(cycles_vec > 0)[0]
    cycles_vec_pos = cycles_vec[cycles_pos_ind].astype(int)
    cum_ind = np.insert(np.cumsum(cycles_vec_pos), 0, 0)

    # total number of 3-cycles (triangles)
    num_cycles = cum_ind[-1]
    
    # store the relative rotations between two nodes in RijMat4d
    RijMat4d = np.zeros((3,3,num_nodes,num_nodes))
    IndMat = np.zeros((num_nodes,num_nodes))

    ### construct edge index matrix
    for l in range(num_edges):
        i, j = Ind_i[l], Ind_j[l]

        # store the relative rotations between two nodes i and j
        RijMat4d[:,:,i,j] = RijMat[:,:,l]
        RijMat4d[:,:,j,i] = RijMat[:,:,l].T

        # store which edge l connects i and j
        IndMat[i,j] = l
        IndMat[j,i] = l
    
    # ij, jk, ki edge trios of each triangle in the graph
    Ind_ij = np.zeros(num_cycles).astype(int)
    Ind_jk = np.zeros(num_cycles).astype(int)
    Ind_ki = np.zeros(num_cycles).astype(int)
    
    # hold the relative rotations of the jk and ki edges
    Rjk0Mat = np.zeros((3,3,num_cycles))
    Rki0Mat = np.zeros((3,3,num_cycles))
    
    # number of ij edges (one ij edge could be a part of many triangles)
    m_pos = len(cycles_pos_ind)
    for l in range(m_pos):
        # get the index of the lth ij edge in the set of all edges
        IJ = cycles_pos_ind[l]

        # get the two nodes the lth ij edge connects
        i, j = Ind_i[IJ], Ind_j[IJ]

        # find all nodes that connect to both i and j (all possible k nodes)
        CoInd_ij = np.where(AdjMat[i] * AdjMat[j])[0]

        # stores the index of the corresponding edge in the set of all edges
        Ind_ij[cum_ind[l]:cum_ind[l+1]] = IJ        # IJ is the index of edge ij in set of all edges Ind

        # cum_ind[l+1] - cum_ind[l] = # of triangles that has edge ij = size of CoInd_ij (all possible k nodes)
        Ind_jk[cum_ind[l]:cum_ind[l+1]] = IndMat[j,CoInd_ij]        # store the indices of all edges jk in set of all edges Ind in Ind_jk
        Ind_ki[cum_ind[l]:cum_ind[l+1]] = IndMat[CoInd_ij,i]        # store the indices of all edges ki in set of all edges Ind in Ind_ki

        # store the relative matrices
        Rjk0Mat[:,:,cum_ind[l]:cum_ind[l+1]] = RijMat4d[:,:,j,CoInd_ij]
        Rki0Mat[:,:,cum_ind[l]:cum_ind[l+1]] = RijMat4d[:,:,CoInd_ij,i]
    

    Rij0Mat = RijMat[:,:,Ind_ij.astype(int)]

    print("compute R cycle")

    # use np.einsum to make use of parallel computation
    R_cycle0 = np.einsum('ijk,jlk->ilk', Rij0Mat, Rjk0Mat)
    R_cycle = np.einsum('ijk,jlk->ilk', R_cycle0, Rki0Mat)

    R_trace = np.clip(R_cycle[0,0] + R_cycle[1,1] + R_cycle[2,2], -1, 3)
    S0_long = abs(np.arccos((R_trace-1) / 2.0)) / np.pi # computing geodesic distance from the ground truth

    S0_vec = np.ones(num_edges)
    
    Weight_vec = np.ones(num_cycles)
    S0_weight = S0_long * Weight_vec
    
    for l in range(m_pos):
        IJ = cycles_pos_ind[l]
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
            IJ = cycles_pos_ind[l]
            weighted_sum = np.sum(S0_weight[cum_ind[l]:cum_ind[l+1]])
            weight_total = np.sum(Weight_vec[cum_ind[l]:cum_ind[l+1]])
            SVec[IJ] = weighted_sum / weight_total

        # parameter controling the decay rate of reweighting function
        beta = beta*rate
        
        print('Reweighting Iteration {} Completed!'.format(iter))   
        iter += 1

    return SVec
