"""
My Model 2 - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
# Import packages
# General
import time
import numpy as np
# My code
import pars_shocks as ps
from pars_shocks import Pars, Shocks
import model_uncert as model
import my_toolbox as tb
import solver
import simulate as simulate
import plot_lc as plot_lc
import run 
import io_manager as io

    
def main_io( H_trans_ind: int = 0, out_folder_name: str = None, H_trans_path: str = None):
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    if out_folder_name is not None:
        print(f"*****Running main_io with out_folder_name = {out_folder_name}*****")
    else:
        print(f"*****Running main_io with default out_folder_name*****")

    # my_lab_FE_grid = np.array([10.0, 15.0, 20.0, 25.0])
    my_lab_FE_grid = np.array([5.0, 10.0, 15.0, 20.0])
    # my_lab_FE_grid = np.array([5.0, 10.0, 15.0])
    my_lab_FE_grid = np.log(my_lab_FE_grid)
    lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
    quad_wage_coeffs = [-0.000, -0.02, -0.02, -0.02] 
    cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

    num_FE_types = len(my_lab_FE_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])

    
    w_coeff_grid[0, :] = [my_lab_FE_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
    w_coeff_grid[1, :] = [my_lab_FE_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
    w_coeff_grid[2, :] = [my_lab_FE_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
    w_coeff_grid[3, :] = [my_lab_FE_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

    print("intial wage coeff grid")
    print(w_coeff_grid)

    my_lab_FE_weights = tb.gen_even_weights(w_coeff_grid)

    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), H_weights=np.array([0.5, 0.5]),
                nu_grid_size=1, alpha = 0.45, sim_draws=1000, lab_FE_grid = my_lab_FE_grid, lab_FE_weights = my_lab_FE_weights,
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 30, sigma_util = 0.9999,
                print_screen=0)
    
    # Get population shares
    k_means_path = main_path + "input/k-means/"
    pop_share_path = k_means_path + "MH_clust_k2_pop_shares.csv"
    beg_pop_weights = tb.read_specific_row_from_csv(pop_share_path, 0)[myPars.H_type_perm_grid_size:].reshape(myPars.H_type_perm_grid_size, myPars.H_grid_size)
    type_pop_share = tb.read_matrix_from_csv(pop_share_path, column_index = 0)[:myPars.H_type_perm_grid_size]
    myPars.H_beg_pop_weights_by_H_type = beg_pop_weights
    myPars.H_type_perm_weights = type_pop_share

    # Get health transition matrix
    out_path = None
    if out_folder_name is not None:
        out_path = myPars.path + out_folder_name + '/'
    if H_trans_path is not None:
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars, path = H_trans_path)
    else:
        print("Using default health transition matrix")
    
    print(f"Age {myPars.age_grid[0]} health transitions:")
    print(myPars.H_trans[:,0,:,:])
    myShocks = Shocks(myPars)

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, 
                          get_moments = True, output_flag = True, tex = True, output_path = out_path)


#run stuff here
start_time = time.perf_counter()
print("Running main")
main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

# trans_path = main_path + "input/MH_trans/MH_trans_by_MH_clust_k2_age.csv"
trans_path = main_path + "input/k-means/MH_trans_by_MH_clust_age.csv"
# of_name = "output"
of_name = None
main_io(out_folder_name = of_name, H_trans_path = trans_path)

tb.print_exec_time("Main.py executed in", start_time) 