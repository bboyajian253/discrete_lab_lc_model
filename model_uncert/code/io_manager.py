"""
io_manager.py
Created on 2024-08-12 14:47:07 

by @author Ben Boyajian

"""
# imports
# general
import numpy as np
import csv 
import os
from typing import Dict, Tuple

# my code
import my_toolbox as tb
from pars_shocks import Pars, Shocks

def get_H_type_pop_shares(myPars: Pars, input_csv_path: str)-> Tuple[np.ndarray, np.ndarray]:
    """
    read in data for myPars.H_beg_pop_weights_by_H_type and myPars.H_type_perm_weights from input_csv_path
    return H_beg_pop_weights, type_pop_share
    """
    pop_share_path = input_csv_path
    H_beg_pop_weights = tb.read_specific_row_from_csv(pop_share_path, 0)[myPars.H_type_perm_grid_size:].reshape(myPars.H_type_perm_grid_size, myPars.H_grid_size)
    type_pop_share = tb.read_matrix_from_csv(pop_share_path, column_index = 0)[:myPars.H_type_perm_grid_size]
    return H_beg_pop_weights, type_pop_share

def get_H_trans_matrix(myPars: Pars, input_csv_path: str)-> np.ndarray:
    """
    read in data for myPars.H_trans from input_csv_path
    *we may not need this function or we can write one that accounts for all the possible different shapes of
    *the health transition matrix using the functions already written in io_manager
    """
    pass

def read_and_shape_H_trans_full(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the full by age by type transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_by_MH_clust_age.csv"
    raw_mat = tb.read_matrix_from_csv(path)
    H_trans_mat_size_j = myPars.H_grid_size * myPars.H_grid_size   
    mat_sep_groups = raw_mat.reshape(myPars.J,  H_trans_mat_size_j, myPars.H_type_perm_grid_size) 
    final_reshape = mat_sep_groups.reshape(myPars.J, myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size).transpose(1, 0, 2, 3)
    return final_reshape

def read_and_shape_H_trans_uncond_age(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the transition matrix for the health state conditional on age but not type
    """
    if path is None:
        path = myPars.path + "input/MH_trans_uncond_age.csv"
    raw_mat = tb.read_matrix_from_csv(path)
    reshaped_mat = raw_mat.reshape(myPars.J, myPars.H_grid_size, myPars.H_grid_size) # Ensure mat is a 3D array
    repeated_mat = np.repeat(np.array(reshaped_mat)[np.newaxis, :, :, :], myPars.H_type_perm_grid_size, axis = 0)
    return repeated_mat

def read_and_shape_H_trans_uncond(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the unconditional on type and age transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_uncond.csv"
    mat = tb.read_specific_row_from_csv(path, 0)
    mat = np.array(mat).reshape(myPars.H_grid_size, myPars.H_grid_size)  # Ensure mat is a 2D array
    repeated_matrix = np.tile(mat, (2, 1, 1))
    H_trans = np.repeat(np.array(repeated_matrix)[:, np.newaxis, :,:], myPars.J, axis=0).reshape(myPars.H_type_perm_grid_size, myPars.J,
                                                                                                 myPars.H_grid_size, myPars.H_grid_size)
    return H_trans

def read_and_shape_H_trans_H_type(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the by type transition matrix (UNCONDITIONAL ON AGE) for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_by_MH_clust.csv"
    raw_mat = tb.read_specific_row_from_csv(path, 0)
    raw_mat = np.array(raw_mat).reshape(myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size)  # Ensure mat is a 3D array
    H_trans = np.repeat(np.array(raw_mat)[:, np.newaxis, :,:], myPars.J, axis=0).reshape(myPars.H_type_perm_grid_size,myPars.J,
                                                                                         myPars.H_grid_size, myPars.H_grid_size)
    return H_trans

def print_params_to_csv(myPars: Pars, path: str = None, file_name: str = "parameters.csv")-> None:
    """
    prints the parametes from myPars to a csv file
    takes in the path and file name with a .csv extension
    """
    if path is None:
        path = myPars.path + 'output/calibration/'
    else:
        path = path + 'calibration/'
    if not os.path.exists(path):
        os.makedirs(path)
    my_path = path + file_name
    with open(my_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        for param, value in pars_to_dict(myPars).items():
            writer.writerow([param, value])

def pars_to_dict(pars_instance: Pars) -> Dict:
    return {
        'lab_fe_grid': pars_instance.lab_fe_grid,
        'lab_fe_grid_size': pars_instance.lab_fe_grid_size,
        'lab_fe_weights': pars_instance.lab_fe_weights,
        'beta': pars_instance.beta,
        'alpha': pars_instance.alpha,
        'sigma_util': pars_instance.sigma_util,
        'phi_n': pars_instance.phi_n,
        'phi_H': pars_instance.phi_H,
        'r': pars_instance.r,
        'a_min': pars_instance.a_min,
        'a_max': pars_instance.a_max,
        'a_grid_growth': pars_instance.a_grid_growth,
        'a_grid': pars_instance.a_grid,
        'a_grid_size': pars_instance.a_grid_size,
        'H_type_perm_grid': pars_instance.H_type_perm_grid,
        'H_type_perm_grid_size': pars_instance.H_type_perm_grid_size,
        'H_type_perm_weights': pars_instance.H_type_perm_weights,
        'H_beg_pop_weights_by_H_type': pars_instance.H_beg_pop_weights_by_H_type,
        'H_grid': pars_instance.H_grid,
        'H_grid_size': pars_instance.H_grid_size,
        'H_trans': pars_instance.H_trans,
        'state_space_shape': pars_instance.state_space_shape,
        'state_space_shape_no_j': pars_instance.state_space_shape_no_j,
        'state_space_no_j_size': pars_instance.state_space_no_j_size,
        'state_space_shape_sims': pars_instance.state_space_shape_sims,
        'lab_min': pars_instance.lab_min,
        'lab_max': pars_instance.lab_max,
        'c_min': pars_instance.c_min,
        'leis_min': pars_instance.leis_min,
        'leis_max': pars_instance.leis_max,
        'sim_draws': pars_instance.sim_draws,
        'J': pars_instance.J,
        'print_screen': pars_instance.print_screen,
        'interp_eval_points': pars_instance.interp_eval_points,
        'sim_interp_grid_spec': pars_instance.sim_interp_grid_spec,
        'start_age': pars_instance.start_age,
        'end_age': pars_instance.end_age,
        'age_grid': pars_instance.age_grid,
        'path': pars_instance.path,
        'wage_coeff_grid': pars_instance.wage_coeff_grid,
        'wH_coeff': pars_instance.wH_coeff,
        'wage_min': pars_instance.wage_min,
        'max_iters': pars_instance.max_iters,
        'max_calib_iters': pars_instance.max_calib_iters
    }

# run if main function
if __name__ == "__main__":

    path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    myPars = Pars(path, J=51)

