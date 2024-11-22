"""
MH_trans_manager.py
Created on 2024-10-21 13:41:14

by @author Ben Boyajian
"""
# imports
# general
import numpy as np
import csv 
import os
from typing import Dict, Tuple
from numba import njit

# my code
import my_toolbox as tb
from pars_shocks import Pars, Shocks

def MH_trans_to_np(myPars: Pars, trans_path: str) -> np.ndarray:
    """
    Read in health transitions from path and return as numpy array
    designed for MH_trans_uncond_age.csv dimensionality
    """
    raw_mat = tb.read_matrix_from_csv(trans_path)
    reshaped_mat = raw_mat.reshape(myPars.J, myPars.H_grid_size, myPars.H_grid_size) # Ensure mat is a 3D array
    return reshaped_mat

@njit
def calc_full_MH_trans(myPars: Pars, trans_reshaped: np.ndarray) -> np.ndarray:
    """
    Calculate full health transition matrix from reshaped matrix with shape (J+1, H_grid_size, H_grid_size)
    """
    ret_mat = np.zeros((myPars.H_type_perm_grid_size, myPars.J, myPars.H_grid_size, myPars.H_grid_size))
    mat_BB = trans_reshaped[:, 0, 0]
    mat_GG = trans_reshaped[:, 1, 1]

    # it would be straightforward to change this to mat_BB*(1+delta) and mat_GG*(1+delta) if we wanted to
    mat_BB_low_typ = mat_BB + myPars.delta_pi_BB
    mat_GG_low_typ = mat_GG - myPars.delta_pi_GG
    mat_BB_high_typ = mat_BB - myPars.delta_pi_BB
    mat_GG_high_typ = mat_GG + myPars.delta_pi_GG

    ret_mat[0, :, 0, 0] = mat_BB_low_typ
    ret_mat[0, :, 0, 1] = 1 - mat_BB_low_typ 
    ret_mat[0, :, 1, 0] = 1 - mat_GG_low_typ 
    ret_mat[0, :, 1, 1] = mat_GG_low_typ

    ret_mat[1, :, 0, 0] = mat_BB_high_typ
    ret_mat[1, :, 0, 1] = 1 - mat_BB_high_typ
    ret_mat[1, :, 1, 0] = 1 - mat_GG_high_typ
    ret_mat[1, :, 1, 1] = mat_GG_high_typ

    return ret_mat

if __name__ == "__main__":
    pass