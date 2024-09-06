"""
model_uncert - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
# Import packages
# General
import time
import numpy as np
from numba import njit
from typing import Tuple, Dict
# My code
import pars_shocks 
from pars_shocks import Pars, Shocks
import model_uncert as model
import my_toolbox as tb
import solver
import simulate as simulate
import plot_lc as plot_lc
import run 
import io_manager as io

    
def main_io(main_path: str, out_folder_name: str = None, H_trans_path: str = None, H_type_pop_share_path: str = None
            ) -> Tuple[Pars, Shocks, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    run the model with the given parameters and return myPars, myShocks, sols, sims
    """

    if out_folder_name is not None:
        print(f"*****Running main_io with out_folder_name = {out_folder_name}*****")
    else:
        print(f"*****Running main_io with default out_folder_name*****")

    # Set wage coefficients
    my_lab_fe_grid = np.log(np.array([10.0, 15.0, 20.0, 25.0]))
    w_coeff_grid = pars_shocks.gen_default_wage_coeffs(my_lab_fe_grid)
    print("intial wage coeff grid", w_coeff_grid)
    # Initialize parameters
    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), 
                alpha = 0.45, sim_draws=1000, lab_fe_grid = my_lab_fe_grid, lab_fe_weights =  tb.gen_even_row_weights(w_coeff_grid),
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 15, sigma_util = 0.9999,
                print_screen=0)
   # Get and set some parameters 
    if H_type_pop_share_path is None:
        H_type_pop_share_path = main_path + "input/k-means/" + "MH_clust_k2_pop_shares.csv"
    myPars.H_beg_pop_weights_by_H_type, myPars.H_type_perm_weights = io.get_H_type_pop_shares(myPars, H_type_pop_share_path)
    if H_trans_path is not None:
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars, path = H_trans_path)
    else:
        print("Using default health transition matrix")
    myShocks = Shocks(myPars)
    
    print(f"Age {myPars.age_grid[0]} health transitions:")
    print(myPars.H_trans[:,0,:,:])
    out_path = None
    if out_folder_name is not None:
        out_path = myPars.path + out_folder_name + '/'

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, 
                          get_targets = True, output_flag = True, tex = True, output_path = out_path)
    return myPars, myShocks, sols, sims
# run if main condition
if __name__ == "__main__":
    #run stuff here
    start_time = time.perf_counter()
    print("Running main")

    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    trans_path = main_path + "input/k-means/MH_trans_by_MH_clust_age.csv"
    of_name = None
    main_io(main_path, out_folder_name = of_name, H_trans_path = trans_path)

    tb.print_exec_time("Main.py executed in", start_time) 