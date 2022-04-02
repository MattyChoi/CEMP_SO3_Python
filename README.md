# Robust Group Synchronization via Cycle-Edge Message Passing (CEMP)

## CEMP Has Broad Applications

Cycle-edge message passing (CEMP) is a very useful algorithm for robust group synchronization (GS). Examples of GS problem include ``correlation clustering`` (Z2 group), ``phase/angular synchronization`` (SO(2) group), ``rotation averaging`` (SO(3) group), and ``multi-object matching`` (Sn group).

The GS problem asks to recover group elements <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{i}^*}"> (star means ground truth) from their noisy/corrupted relative measurements <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{ij}}"> (ideally equals to <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{i}^*g_{j}^{*-1}}">).

CEMP not only classifies the clean and corrupted relative measurements (group ratios), but also measures their corruption levels. That is, for each edge (i,j), CEMP estimates the distance of <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{ij}}"> from its ground truth. The following is a typical scatter plot (when ``70%`` of edges are corrupted) of CEMP-estimated corruption levels v.s the ground truth ones, indicating exact estimation (align well with the line y=x).

<img src="https://github.com/yunpeng-shi/CEMP/blob/main/scatter.jpg" width="500" height="400">

## CEMP Offers Various Ways for Solving Group Sync
After estimating corruption levels <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{s_{ij}^* = d(g_{ij}, g_{ij}^*)}">, there are two primary ways to estimate the absolute group elements <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{i}^*}">:

First, one can build a weighted graph using the estimated corruption levels, and find its minimum spanning tree (MST) so that it's the cleanest spanning tree. Then, one can fix the first group element as identity, and find the rest of them by subsequently multiplying group ratios along the spanning tree. It is called ``CEMP+MST``

Second, one can implement a weighted spectral method (that approximately solves a weighted least squares problem) where the weights focuses on the clean edges. It is called ``CEMP+GCW``. Here GCW refers to the fact that we do spectral decomposition on the graph connection weight matrix.

See details in
[Robust Group Synchronization via Cycle-Edge Message Passing](https://link.springer.com/content/pdf/10.1007/s10208-021-09532-w.pdf), Gilad Lerman and Yunpeng Shi, Foundations of Computational Mathematics, 2021.

For other possible usage of CEMP, see repo (https://github.com/yunpeng-shi/MPLS) and (https://github.com/yunpeng-shi/IRGCL).

## A variety of Groups and Metrics
[``Z2``](https://github.com/yunpeng-shi/CEMP/tree/main/Z2) folder is for Z2-synchronization with applications in correlation clustering.

[``SO2``](https://github.com/yunpeng-shi/CEMP/tree/main/SO2) folder is for angular synchronization (SO(2) group). The metric of CEMP is chosen as geodesic distance in U(1).

[``SO3``](https://github.com/yunpeng-shi/CEMP/tree/main/SO3) folder is for rotation synchronization (SO(3) group), or rotation averaging. The metric of CEMP is chosen as geodesic distance in SO(3).

[``SOd``](https://github.com/yunpeng-shi/CEMP/tree/main/SOd) folder is for general SO(d) synchronization. The metric of CEMP is chosen as the difference in Frobenius norm.

[``MPLS``](https://github.com/yunpeng-shi/MPLS) repository offers a faster implementation of CEMP-SO(3) with sampled 3-cycles (not like this repo that uses all 3-cycles). It also includes the state-of-the-art rotation averaging method MPLS.

[``IRGCL``](https://github.com/yunpeng-shi/IRGCL) repository offers a fully-vectorized version of CEMP-Sn. It aims to solve the permutation synchronization/multi-object matching problem. It also includes a MPLS-like algorithm (called IRGCL) for permutation sync.


## A Variety of Algorithms to Compare

In most of above folders, we include in ``Algorithms`` subfolder that contains the implementation of the following methods.

``Spectral`` refers to eigenvector method for approximately solving least squares formulation. See [Angular Synchronization by Eigenvectors and Semidefinite Programming,](https://arxiv.org/abs/0905.3174) Amit Singer, Applied and Computational Harmonic Analysis, 2011 for details.

``SDP`` refers to semi-definite relaxation method for approximately solving least squares formulation. See [Angular Synchronization by Eigenvectors and Semidefinite Programming,](https://arxiv.org/abs/0905.3174) Amit Singer, Applied and Computational Harmonic Analysis, 2011 for details.

``IRLS`` refers to iteratively reweighted least squares (IRLS) that uses L1 loss function. It iteratively solves a weighted spectral methods, where the edge weights are updated as the reciprocal of the residuals.

``CEMP+MST``, ``CEMP+GCW`` refer to our two post-processing methods after implementing CEMP. See [Robust Group Synchronization via Cycle-Edge Message Passing](https://link.springer.com/content/pdf/10.1007/s10208-021-09532-w.pdf), Gilad Lerman and Yunpeng Shi, Foundations of Computational Mathematics, 2021 for details.


## A Variety of Corruption Models
We provide 5 different corruption models. 3 for nonuniform topology and 2 for uniform toplogy (see ``Uniform_Topology.m`` and ``Nonuniform_Topology.m``). Uniform/Nonuniform toplogy refers to whether the corrupted subgraph is Erdos Renyi or not. In other words, the choice of Uniform/Nonuniform toplogy decides how to select edges for corruption. In ``Uniform_Topology.m``, two nodes are connected with probability ``p``. Then edges are independently drawn with probability ``q`` for corruption. In ``Nonuniform_Topology.m``, two nodes are connected with probability ``p``. Then with probability ``p_node_crpt`` a node is selected so that its neighboring edges are candidates for corruption. Next, for each selected node, with probability ``p_edge_crpt`` an edge (among the neighboring edges of the selected node) is corrupted. This is a more malicious scenario where corrupted edges have cluster behavior (so local coruption level can be extremely high). 

One can also optionally add noise to inlier graph and outlier graph through setting ``sigma_in`` and ``sigma_out`` for ``Nonuniform_Topology.m``. For ``Uniform_Topology.m`` we assume inlier and outlier subgraph have the same noise level ``sigma``.

The argument ``crpt_type`` in the two functions determines how the corrupted group ratios are generated for those selected edges. In ``Uniform_Topology.m``, there are 2 options of ``crpt_type``: ``uniform`` and ``self-consistent``.
In ``Nonuniform_Topology.m``, there are the following 3 options of ``crpt_type``.

``uniform``: The corrupted group ratios <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{ij}}"> are i.i.d follows uniform distribution over the space of the group.

``self-consistent``: The corrupted <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{ij}}"> are group ratios of another set of absolute rotations. Namely <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{ij} = g_i^{crpt} g_j^{crpt}'}"> where those absolute group elements are different from the ground truth and are i.i.d drawn from the uniform distribution in the space of the group. In this way, the corrupted group ratios are also cycle-consistent.

``adv``: Extremely malicious corruption that replaces the underlying absolute group elements from ground truth <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_i^*}"> to <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_i^{crpt}}">. Namely <img src="https://render.githubusercontent.com/render/math?math=\color{red} \mathbf{g_{ij} = g_i^{crpt} g_j^{* }'}"> for the corrupted neighboring edges (i,j) of node i. Additional high noise must be added to the outlier-subgraph, otherwise the recovery of the ground truth can be ill-posed. It was first introduced in [Robust Multi-object Matching via Iterative Reweighting of the Graph Connection Laplacian, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/ae06fbdc519bddaa88aa1b24bace4500-Paper.pdf) for permutation synchronization.



## Implementation of CEMP

The demo code in each group folder uses the following function for implementing CEMP:
```
CEMP(Ind, RijMat, parameters)
```
Each row of ``Ind`` matrix is an edge index (i,j). The edge indices (the rows of Ind) MUST be sorted in ``row-major order``. That is, the edge indices are sorted as  for example (1,2), (1,3), (1,4),..., (2,3), (2,5), (2,8),..., otherwise the code may crash when some edges are not contained in any 3-cycles. Make sure that i<j. If some edges have indices (3,1), then change it to (1,3) and take a transpose to the corresponding Rij. See also ``Examples/Compare_algorithms.m`` in each subfolder of groups for details.

## Dependencies
The implementation of SDP relaxation requires [CVX](http://cvxr.com/cvx/) package. If CVX are not (or cannot be) installed, simply comment out the lines that runs SDP in ``Examples/Compare_algorithms.m``. Note that our methods do not rely on CVX. It is only for comparing with other baseline methods.



