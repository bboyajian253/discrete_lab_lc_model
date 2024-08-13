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
from pars_shocks import Pars

def read_and_shape_h_trans(myPars: Pars) -> np.ndarray:
    """
    Read in the transition matrix for the health state and reshape it to the correct dimensions
    """
    path = myPars.path + "input/MH_trans.csv"
    # Read in the data
    all_trans = tb.read_matrix_from_csv(path).reshape(myPars.H_type_perm_grid.size, myPars.J, myPars.H_grid_size, myPars.H_grid_size)
    # Reshape
    return all_trans

# run if main function
if __name__ == "__main__":
    path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    myPars = Pars(path)

    trans = read_and_shape_h_trans(myPars)
    print(trans)
    print(trans.shape)
