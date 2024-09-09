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

# move to factory class?
def pars_factory(main_path: str, H_trans_path: str = None, H_type_pop_share_path: str = None) -> Pars:
    """
    create and returh a pars object with default parameters
    """
    # Set wage coefficients
    my_lab_fe_grid = np.log(np.array([5.0, 10.0, 15.0, 20.0]))
    w_coeff_grid = pars_shocks.gen_default_wage_coeffs(my_lab_fe_grid)
    # Initialize parameters
    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), 
                alpha = 0.45, sim_draws=1000, lab_fe_grid = my_lab_fe_grid, lab_fe_weights =  tb.gen_even_row_weights(w_coeff_grid),
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 100, sigma_util = 0.9999,
                print_screen=0)
    # Get and set some parameters 
    if H_type_pop_share_path is None:
        H_type_pop_share_path = main_path + "input/k-means/" + "MH_clust_k2_pop_shares.csv"
    myPars.H_beg_pop_weights_by_H_type, myPars.H_type_perm_weights = io.get_H_type_pop_shares(myPars, H_type_pop_share_path)
    if H_trans_path is not None:
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars, path = H_trans_path)
    else:
        print("Using default health transition matrix")
    return myPars
    
def main_io(main_path: str, myPars: Pars = None, myShocks: Shocks = None, out_folder_name: str = None, H_trans_path: str = None, H_type_pop_share_path: str = None, 
            output_flag: bool = True, do_wH_calib: bool = True) -> Tuple[Pars, Shocks, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    run the model with the given parameters and return myPars, myShocks, sols, sims
    """
    # myPars = givPars 

    if out_folder_name is not None:
        print(f"*****Running main_io with out_folder_name = {out_folder_name}*****")
    else:
        print(f"*****Running main_io with default out_folder_name*****")
    if myPars is None:
        myPars = pars_factory(main_path = main_path, H_trans_path= H_trans_path, H_type_pop_share_path = H_type_pop_share_path)
    if myShocks is None:
        myShocks = Shocks(myPars)
    out_path = None
    if out_folder_name is not None:
        out_path = myPars.path + out_folder_name + '/'

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, do_wH_calib = do_wH_calib, sim_no_calib = False, 
                          get_targets = True, output_flag = output_flag, tex = True, output_path = out_path)
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