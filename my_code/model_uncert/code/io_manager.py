"""
pars_shocks_and_wages.py

Created on 2024-08-12 14:47:07 

by @author Ben Boyajian

"""
# imports
# general
import numpy as np
import csv 
# my code
import my_toolbox as tb
import pars_shocks as ps
from pars_shocks import Pars, Shocks

def read_and_shape_H_trans_full(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_by_MH_clust_age.csv"
    # Read in the data
    raw_mat = tb.read_matrix_from_csv(path)
    H_trans_mat_size_j = myPars.H_grid_size * myPars.H_grid_size   
    mat_sep_groups = raw_mat.reshape(myPars.J,  H_trans_mat_size_j, myPars.H_type_perm_grid_size) 
    # print(f"mat_sep_groups: {mat_sep_groups}")
    # Transpose the axes to match the desired structu
    # # Now reorder the reshaped matrix to (2, 51, 2, 2) structure
    final_reshape = mat_sep_groups.reshape(myPars.J, myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size).transpose(1, 0, 2, 3)

    return final_reshape

def read_and_shape_H_trans_uncond(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the unconditional transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_uncond.csv"
    # Read in the data
    mat = tb.read_specific_row_from_csv(path, 0)
    mat = np.array(mat).reshape(myPars.H_grid_size, myPars.H_grid_size)  # Ensure mat is a 2D array
    # Repeat the matrix
    repeated_matrix = np.tile(mat, (2, 1, 1))
    H_trans = np.repeat(np.array(repeated_matrix)[:, np.newaxis, :,:], 51, axis=0).reshape(2,51,2,2)
    return H_trans

def read_and_shape_H_trans_H_type(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_by_MH_clust.csv"
    # Read in the data
    raw_mat = tb.read_specific_row_from_csv(path, 0)
    raw_mat = np.array(raw_mat).reshape(myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size)  # Ensure mat is a 3D array
    # Repeat the matrix
    H_trans = np.repeat(np.array(raw_mat)[:, np.newaxis, :,:], 51, axis=0).reshape(2,51,2,2)
    return H_trans




# run if main function
if __name__ == "__main__":

    path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    myPars = Pars(path, J=51)

    trans_uncond = read_and_shape_H_trans_uncond(myPars)
    trans_H_type = read_and_shape_H_trans_H_type(myPars)
    trans_full = read_and_shape_H_trans_full(myPars)
    print(f"trans_H_type = {trans_H_type}")
    print(f"trans_uncond.shape = {trans_uncond.shape}")
    print(f"trans_H_type.shape = {trans_H_type.shape}")
    print(f"trans_full.shape = {trans_full.shape}")

