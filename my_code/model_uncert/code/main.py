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
import pandas as pd
import MH_trans_manager as trans_manager

# move to factory class?
def pars_factory(main_path: str, H_trans_uncond_path: str = None, H_type_pop_share_path: str = None, my_lab_fe_grid: np.ndarray = None
                 ) -> Pars:
    """
    create and returh a pars object with default parameters
    """
    # Set wage coefficients
    if my_lab_fe_grid is None:
        # my_lab_fe_grid = np.log(np.array([5.0, 10.0, 15.0, 20.0]))
        my_grid = np.linspace(5.0, 20.0, 10) 
        my_grid = np.log(my_grid)
        my_lab_fe_grid = my_grid

    w_coeff_grid = pars_shocks.gen_default_wage_coeffs(my_lab_fe_grid)
    # Initialize parameters
    myPars = Pars(main_path, J=51, a_grid_size=301, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), 
                alpha = 0.45, sim_draws=1000, lab_fe_grid = my_lab_fe_grid, lab_fe_weights =  tb.gen_even_row_weights(w_coeff_grid),
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 100, sigma_util = 0.9999,
                print_screen=0)
    myPars.H_type_perm_weights = np.array([0.5, 0.5])
    share_bad = pd.read_csv(main_path + "input/50p_age_moms/mean_bad_MH_by_age.csv")['mean_badMH'].iloc[0]
    myPars.H_beg_pop_weights_by_H_type = np.array([[share_bad, 1-share_bad], [share_bad, 1-share_bad]]) 
    # # Get and set some parameters 
    # if H_type_pop_share_path is None:

    #     # H_type_pop_share_path = main_path + "input/50p_age_moms/" + "MH_clust_50p_age_pop_shares.csv"
    # myPars.H_beg_pop_weights_by_H_type, myPars.H_type_perm_weights = io.get_H_type_pop_shares(myPars, H_type_pop_share_path)
    if H_trans_uncond_path is not None:
        H_trans_uncond = trans_manager.MH_trans_to_np(myPars, H_trans_uncond_path)
        myPars.set_H_trans_uncond(H_trans_uncond)
    else:
        print("Using default health transition matrix")

    return myPars
    
def main_io(main_path: str, myPars: Pars = None, myShocks: Shocks = None, out_folder_name: str = None, H_trans_uncond_path: str = None, H_type_pop_share_path: str = None, 
            my_lab_fe_grid: np.ndarray = None, output_flag: bool = True, do_wH_calib: bool = True) -> Tuple[Pars, Shocks, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    run the model with the given parameters and return myPars, myShocks, sols, sims
    """
    # myPars = givPars 

    if out_folder_name is not None:
        print(f"*****Running main_io with out_folder_name = {out_folder_name}*****")    
    else:
        print(f"*****Running main_io with default out_folder_name*****")
    if myPars is None:
        myPars = pars_factory(main_path = main_path, H_trans_uncond_path= H_trans_uncond_path, H_type_pop_share_path = H_type_pop_share_path, my_lab_fe_grid = my_lab_fe_grid)
    if myShocks is None:
        myShocks = Shocks(myPars)
    outpath = None
    if out_folder_name is not None:
        outpath = myPars.path + out_folder_name + '/'

    # myPars.print_screen = 2

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, do_wH_calib = do_wH_calib, sim_no_calib = False, 
                          get_targets = True, output_flag = output_flag, tex = True, output_path = outpath, 
                          data_moms_folder_path= myPars.path + '/input/50p_age_moms/')
    return myPars, myShocks, sols, sims

# run if main condition
if __name__ == "__main__":
    import plot_inequality as plot_ineq
    #run stuff here
    start_time = time.perf_counter()
    print("Running main")

    # ***** may want to change how trans is generated its redundant in do file.
    of_name = None
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    trans_path = main_path + "input/50p_age_moms/MH_trans_uncond_age.csv"
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

    myPars, myShocks, sols, sims = main_io(main_path, out_folder_name = of_name, H_trans_uncond_path = trans_path)

    tb.print_exec_time("Main.py executed in", start_time) 