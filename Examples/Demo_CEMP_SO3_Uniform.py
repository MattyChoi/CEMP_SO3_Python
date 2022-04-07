import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../Algorithms')
sys.path.append('../Models')
from CEMP_SO3 import *
from Uniform_Topology_SO3 import *

out = Uniform_Topology_SO3(200, 0.5, 0.3, 0.1, "uniform")
Ind = out.Ind; # matrix of edge indices (m by 2)
RijMat = out.RijMat; # given corrupted and noisy relative rotations
ErrVec = out.ErrVec; # ground truth corruption levels
R_orig = out.R_orig; # ground truth rotations

# set CEMP defult parameters
beta_init = 1
beta_max = 40
rate = 1.2

# run CEMP
SVec = CEMP_SO3(Ind, RijMat, beta_init, beta_max, rate)

#visualize sij^* and sij,t, ideally one should see a straight line "y=x"
plt.plot(ErrVec, SVec, 'b.')
plt.title(r'Scatter Plot of $s_{ij}^*$ vs $s_{ij,T}$')
plt.xlabel(r'$s_{ij}^*$') 
plt.ylabel(r'$s_{ij,T}$')
plt.show() 
