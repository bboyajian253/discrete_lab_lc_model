from pars_and_shocks import Pars
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def manual_kron(a, b):
    m, n = a.shape
    p, q = b.shape
    result = np.zeros((m * p, n * q), dtype=a.dtype)
    for i in prange(m):
        for j in prange(n):
            for k in prange(p):
                for l in prange(q):
                    result[i * p + k, j * q + l] = a[i, j] * b[k, l]
    return result

@njit
def gen_flat_joint_trans(trans1, trans2):
    joint_transition = manual_kron(trans1, trans2)
    return joint_transition.flatten()

def  print_gen_flat_joint_trans() :
    H_grid,nu_grid = myPars.H_grid, myPars.nu_grid
    H_trans, nu_trans = myPars.H_trans,myPars.nu_trans

    print("H_grid", "\n", H_grid) 
    print("H_trans", "\n", H_trans)

    print("nu_grid", "\n", nu_grid) 
    print("nu_trans", "\n", nu_trans)

    joint_transition = np.kron(H_trans, nu_trans)

    print("Joint transition matrix:")
    print(joint_transition)
    print("Shape of joint transition matrix:", joint_transition.shape)

    flattened_joint_transition = joint_transition.flatten()
    print("Flattened joint transition matrix:")
    print(flattened_joint_transition)
    print("Shape of flattened joint transition matrix:", flattened_joint_transition.shape)

 

if __name__ == "__main__" :
    myPars = Pars(print_screen=0, nu_grid_size= 3)

    print_gen_flat_joint_trans()
    print(gen_flat_joint_trans(myPars.H_trans, myPars.nu_trans))


