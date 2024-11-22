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
def pars_factory(main_path: str, H_trans_uncond_path: str = None, H_trans_path: str = None, H_type_pop_share_path: str = None, my_lab_fe_grid: np.ndarray = None, num_sims: int = 1000, 
                 ) -> Pars:
    """
    create and returh a pars object with default parameters
    """
    # Set wage coefficients
    if my_lab_fe_grid is None:
        my_grid = np.linspace(5.0, 20.0, 10) 
        my_grid = np.log(my_grid)
        my_lab_fe_grid = my_grid
    w_coeff_grid = pars_shocks.gen_default_wage_coeffs(my_lab_fe_grid)

    # Initialize parameters
    myPars = Pars(main_path, J=51, a_grid_size=301, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), 
                alpha = 0.45, sim_draws=num_sims, lab_fe_grid = my_lab_fe_grid, lab_fe_weights =  tb.gen_even_row_weights(w_coeff_grid),
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 100, sigma_util = 0.9999,
                print_screen=0)

    # Get and set some parameters 
    # Starting shares of health types
    if H_type_pop_share_path is None:
        myPars.H_type_perm_weights = np.array([0.5, 0.5])
        share_bad = pd.read_csv(main_path + "input/50p_age_moms/mean_bad_MH_by_age.csv")['mean_badMH'].iloc[0]
        myPars.H_beg_pop_weights_by_H_type = np.array([[share_bad, 1-share_bad], [share_bad, 1-share_bad]]) 
    else:
        myPars.H_beg_pop_weights_by_H_type, myPars.H_type_perm_weights = io.get_H_type_pop_shares(myPars, H_type_pop_share_path)

    # Health transition matrix
    if H_trans_path is None and H_trans_uncond_path is None:
        print("Using default health transition matrix")
    else:
        if H_trans_uncond_path is not None:
            H_trans_uncond = trans_manager.MH_trans_to_np(myPars, H_trans_uncond_path)
            myPars.H_trans_uncond = H_trans_uncond
        if H_trans_path is None: 
            # if we want to use H_trans_uncond adjusted by delta_pi if not = 0
            myPars.update_H_trans()
        else: # H_trans_path is not None
            H_trans = io.read_and_shape_H_trans_full(myPars, H_trans_path) 
            myPars.H_trans = H_trans

    return myPars
    
def main_io(main_path: str, myPars: Pars = None, myShocks: Shocks = None, out_folder_name: str = None, 
            H_trans_uncond_path: str = None, H_trans_path:str = None, H_type_pop_share_path: str = None, data_moms_path: str = None,
            my_lab_fe_grid: np.ndarray = None, output_flag: bool = True, num_sims: int = 1000, 
            calib_flag: bool = True, sim_no_calib: bool = False,
            do_wH_calib: bool = True, do_dpi_calib: bool = False, do_phi_H_calib: bool = False, 
            do_eps_gg_calib: bool = True, do_eps_bb_calib: bool = False
            ) -> Tuple[Pars, Shocks, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    run the model with the given parameters and return myPars, myShocks, sols, sims
    """

    if out_folder_name is None:
        print(f"*****Running main_io with default out_folder_name*****")
    else:
        print(f"*****Running main_io with out_folder_name = {out_folder_name}*****")    

    if myPars is None:
        myPars = pars_factory(main_path = main_path, 
                              H_trans_uncond_path= H_trans_uncond_path, H_trans_path=H_trans_path, H_type_pop_share_path = H_type_pop_share_path, 
                              my_lab_fe_grid = my_lab_fe_grid, num_sims = num_sims)
    if myShocks is None:
        myShocks = Shocks(myPars)
    if out_folder_name is None:
        outpath = None
    else:
        outpath = myPars.path + out_folder_name + '/'
    if data_moms_path is None:
        data_moms_path = myPars.path + '/input/50p_age_moms/'

    # myPars.print_screen = 2

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = calib_flag, sim_no_calib = sim_no_calib,  
                                get_targets = True, output_flag = output_flag, tex = True,  
                                do_wH_calib = do_wH_calib,  do_dpi_calib=do_dpi_calib, do_phi_H_calib = do_phi_H_calib, 
                                do_eps_gg_calib = do_eps_gg_calib, do_eps_bb_calib = do_eps_bb_calib,
                                output_path = outpath, data_moms_folder_path = data_moms_path)
    if myShocks is None:
        myShocks = Shocks(myPars)

    return myPars, myShocks, sols, sims

# run if main condition
if __name__ == "__main__":
    import plot_inequality as plot_ineq
    import os
    print("Current working directory:", os.getcwd())

    #run stuff here
    start_time = time.perf_counter()
    print("Running main")

    of_name = None

    main_path = os.path.realpath(__file__ + "/../../") + "/"
    input_path = main_path + "/input/50p_age_moms/"
    trans_path_uncond = input_path + "MH_trans_uncond_age.csv"

    trans_path_50p = input_path + "MH_trans_by_MH_clust_age.csv"
    type_path_50p = input_path + "MH_clust_50p_age_pop_shares.csv"

    myPars, myShocks, sols, sims = main_io(main_path, out_folder_name = of_name, output_flag = True,
                                                H_trans_uncond_path = trans_path_uncond, H_trans_path = trans_path_50p, 
                                                H_type_pop_share_path = type_path_50p,
                                                do_eps_gg_calib=True, do_eps_bb_calib=True)

    tb.print_exec_time("Main.py executed in", start_time) 