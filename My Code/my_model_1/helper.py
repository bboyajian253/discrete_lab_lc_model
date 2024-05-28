from pars_shocks_and_wages import Pars
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


 

if __name__ == "__main__" :
    myPars = Pars(print_screen=0)
    h_ind = 0
    nu_ind = 0
    h_probs = myPars.H_trans[h_ind, :]
    print(h_probs)
    nu_probs = myPars.nu_trans[nu_ind, :]
    print(nu_probs)
    shock_probs = np.outer(h_probs, nu_probs)
    print(shock_probs)
    shock_probs_flat = shock_probs.flatten() 
    print(shock_probs_flat)
