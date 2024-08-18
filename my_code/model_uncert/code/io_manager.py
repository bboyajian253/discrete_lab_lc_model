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

def read_and_shape_h_trans_full(myPars: Pars) -> np.ndarray:
    """
    Read in the transition matrix for the health state and reshape it to the correct dimensions
    """
    path = myPars.path + "input/MH_trans.csv"
    # Read in the data
    raw_mat = tb.read_matrix_from_csv(path)
    # print(f"raw_mat: {raw_mat}")
    # Reshape the matrix into (51, 4, 2) to separate the groups
    H_trans_mat_size_j = myPars.H_grid_size * myPars.H_grid_size   
    mat_sep_groups = raw_mat.reshape(myPars.J,  H_trans_mat_size_j, myPars.H_type_perm_grid_size) 
    print(f"mat_sep_groups: {mat_sep_groups}")
    # Transpose the axes to match the desired structu
    # # Now reorder the reshaped matrix to (2, 51, 2, 2) structure
    final_reshape = mat_sep_groups.reshape(myPars.J, myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size).transpose(1, 0, 2, 3)
    # mat_transpose = mat_sep_groups.transpose(2, 0, 1)
    # print(f"mat_transpose: {mat_transpose}")

    # final_reshape = mat_transpose.reshape(2, 51, 2, 2)
    # print(f"final_reshape: {final_reshape}")

    return final_reshape

# run if main function
if __name__ == "__main__":
    path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    myPars = Pars(path, J=51)

    trans = read_and_shape_h_trans_full(myPars)
    # print(trans)
    print(trans.shape)
    # print(trans[0, :, :, :])

    # myPars.H_trans = trans
    # myShocks = Shocks(myPars)
    # hist = myShocks.H_hist
    # print(hist)
    # print(hist.shape)
    # print("Same pars diff sim")
    # hist0 = hist[0,0,0,0,:]
    # hist1 = hist[0,0,0,1,:]
    # print(hist0)
    # print(np.sum(hist0))
    # print(hist1)
    # print(np.sum(hist1))

