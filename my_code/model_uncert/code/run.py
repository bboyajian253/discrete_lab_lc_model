"""
run.py

This file contains code to solve, simulate, and calibrate the the model depending on the user's choice.

Author: Ben
Date: 2024-06-19 12:07:5

"""

# Import packages
# General
import numpy as np
import time
from typing import List, Dict
import os

# My code
import tables
import my_toolbox as tb
import solver
import simulate as simulate
import calibration
import plot_lc as plot_lc
from pars_shocks import Pars, Shocks
import plot_aggregates as plot_aggregates  
import io_manager as io

# Run the model
def run_model(myPars: Pars, myShocks: Shocks, solve: bool = True, calib : bool = True, do_wH_calib: bool = True, get_targets: bool = True, sim_no_calib  : bool = False, 
              output_flag: bool = True, tex: bool = True, output_path: str = None, data_moms_folder_path: str = None)-> List[Dict[str, np.ndarray]]:
    """
    Given the model parameters, solve, calibrate, simulate, and, if desired, output the results.
    (i) Solve the model
    (ii) Simulate the model
    (iii) Calibrate the model
    (iv) Output the results and model aggregates
    """
    if output_path is None:
        output_path = myPars.path + '/output/'
    if data_moms_folder_path is None:
        data_moms_folder_path = myPars.path + '/input/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if solve:
        start_time = time.perf_counter()
        state_sols = solver.solve_lc(myPars)
        for label, value in state_sols.items():
            np.save(output_path + label + '_lc', value)
        tb.print_exec_time("Solver ran in", start_time)
    
    #always load state specific solutions
    sol_labels = ['c', 'lab', 'a_prime']
    state_sols = {}
    for label in sol_labels:
        state_sols[label] = np.load(output_path + label + '_lc.npy')

    #if no_calibrate_but_sim, simulate without calibrating
    if sim_no_calib:
        start_time = time.perf_counter()
        sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
        for label in sim_lc.keys():
            np.save(output_path + f'sim{label}.npy', sim_lc[label])
        tb.print_exec_time("Simulate ran in", start_time)
    elif calib:
        start_time = time.perf_counter()
        max_iters = myPars.max_iters
        if get_targets: 
            alpha_lab_targ, w0_mu_targ, w0_sigma_targ, w1_targ, w2_targ, wH_targ = calibration.get_all_targets(myPars, target_folder_path=data_moms_folder_path)
            print(f"""Calibrating with alpha_lab_targ = {alpha_lab_targ}, w0_mean_targ = {w0_mu_targ}, w0_sd_targ = {w0_sigma_targ}, w1_targ = {w1_targ}, w2_targ = {w2_targ}, wH_targ = {wH_targ}""")
            calib_alpha, w0_weights, calib_w1, calib_w2, calib_wH, state_sols, sim_lc = calibration.calib_all(myPars, myShocks, do_wH_calib = do_wH_calib,
                                                                                                    alpha_mom_targ = alpha_lab_targ, w0_mu_mom_targ = w0_mu_targ, w0_sigma_mom_targ = w0_sigma_targ, 
                                                                                                    w1_mom_targ = w1_targ, w2_mom_targ = w2_targ, wH_mom_targ = wH_targ)
        else: # otherwise use default argument targets
            calib_alpha, w0_weights, calib_w1, calib_w2, calib_wH, state_sols, sim_lc = calibration.calib_all(myPars, myShocks, do_wH_calib = do_wH_calib)

        calib_targ_vals_dict = { 'alpha': alpha_lab_targ, 
                                'w0_mean': w0_mu_targ, 'w0_sd': w0_sigma_targ, 
                                'w1': w1_targ, 'w2': w2_targ, 
                                'wH': wH_targ}
        calib_model_vals_dict = {   'alpha': calibration.alpha_moment_giv_sims(myPars, sim_lc), 
                                    'w0_mean': calibration.w0_moments(myPars)[0], 'w0_sd': calibration.w0_moments(myPars)[1],
                                    'w1': calibration.w1_moment(myPars), 'w2': calibration.w2_moment(myPars),
                                    'wH': calibration.wH_moment(myPars, myShocks)}

        for label in sim_lc.keys():
            np.save(output_path + f'sim{label}.npy', sim_lc[label])
        tb.print_exec_time("Calibration ran in", start_time)

    #always load simulated life cycles
    sim_labels = ['c', 'lab', 'a', 'wage', 'lab_earnings']
    sim_lc = {}
    for label in sim_labels:
        sim_lc[label] = np.load(output_path + f'sim{label}.npy')

    #if output, output the results
    if output_flag:
        output(myPars, state_sols, sim_lc, calib_targ_vals_dict, calib_model_vals_dict, tex, get_targets, 
               data_moms_folder_path = data_moms_folder_path, out_path = output_path)
    
    return [state_sols, sim_lc]

def output(myPars: Pars, state_sols: Dict[str, np.ndarray], sim_lc: Dict[str, np.ndarray], targ_moments: Dict[str, np.ndarray], 
           model_moments: Dict[str, np.ndarray], tex: bool, get_targets: bool, data_moms_folder_path: str, out_path: str = None)-> None:
    if out_path is None:
        out_path = myPars.path + 'output/tabs_fit_figs/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    io.print_params_to_csv(myPars, out_path)
    if tex:
        tables.print_exog_params_to_tex(myPars, out_path)
        tables.print_endog_params_to_tex(myPars, targ_moments, model_moments, out_path)
        tables.print_w0_calib_to_tex(myPars, targ_moments, model_moments, out_path)
        # calibration.print_H_trans_to_tex(myPars, path)
    if get_targets:
        lab_mom_path = data_moms_folder_path + 'labor_moments.csv'
        plot_aggregates.plot_lab_aggs_and_moms(myPars, sim_lc, data_moms_path=lab_mom_path, out_path = out_path)
        emp_mom_path = data_moms_folder_path + 'emp_rate_moments.csv'
        plot_aggregates.plot_emp_aggs_and_moms(myPars, sim_lc, data_moms_path=emp_mom_path, out_path = out_path)
        wage_mom_path = data_moms_folder_path + 'wage_moments.csv'
        plot_aggregates.plot_wage_aggs_and_moms(myPars, data_moms_path=wage_mom_path, out_path = out_path)
    plot_lc.plot_lc_profiles(myPars, sim_lc, out_path)

# run if main function
if __name__ == "__main__":
  pass 