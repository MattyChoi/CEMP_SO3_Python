import numpy as np
import numpy.linalg as LA

def generate_rotations(length):
    mats = np.zeros((3,3,length))

    for i in range(length):
        Q = np.random.randn(3,3)

        # use svd to get unitary 3x3 matrices
        U, _, V = LA.svd(Q)

        # email yunpeng about this line
        S0 = np.diag([1, 1, LA.det(U @ V)])  
        mats[:,:,i] = U @ S0 @ V
    
    return mats


def project_to_SO3(mats):
    # use svd to get unitary 3x3 matrices
    U, _, V = LA.svd(mats)

    # email yunpeng about this line
    S0 = np.diag([1, 1, LA.det(U @ V)])  
    mats = U @ S0 @ V

    return mats


def project_to_SO3_all(mats, length):
    for i in range(length):
        # use svd to get unitary 3x3 matrices
        U, _, V = LA.svd(mats[:,:,i])

        # email yunpeng about this line
        S0 = np.diag([1, 1, LA.det(U @ V)])  
        mats[:,:,i] = U @ S0 @ V