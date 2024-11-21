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
import my_tables as tables
import my_toolbox as tb
import solver
import simulate as simulate
import calibration
import plot_lc as plot_lc
from pars_shocks import Pars, Shocks
import plot_aggregates as plot_aggregates  
import io_manager as io

# Run the model
def run_model(myPars: Pars, myShocks: Shocks, modify_shocks: bool = True, solve: bool = True,
              calib : bool = True, do_wH_calib: bool = True, do_dpi_calib: bool = True, 
              do_eps_gg_calib: bool = True, do_eps_bb_calib: bool = False,
              do_phi_H_calib: bool = False, get_targets: bool = True, sim_no_calib  : bool = False, 
              output_flag: bool = True, tex: bool = True, output_path: str = None, data_moms_folder_path: str = None)-> List[Dict[str, np.ndarray]]:
    """
    Given the model parameters, solve, calibrate, simulate, and, if desired, output the results.
    (i) Solve the model
    (ii) Simulate the model
    (iii) Calibrate the model
    (iv) Output the results and model aggregates
    """
    if output_path is None:
        output_path = myPars.path + 'output/'
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
            # alpha_lab_targ, w0_mu_targ, w0_sigma_targ, w1_targ, w2_targ, wH_targ, phi_H_targ, dpi_BB_targ, dpi_GG_targ, eps_gg_targ, eps_bb_targ = calibration.get_all_targets(myPars, target_folder_path=data_moms_folder_path)
            calib_targ_vals_dict = calibration.get_all_targets(myPars, target_folder_path=data_moms_folder_path)
            print("Calibrating with targets: ")
            for key, value in calib_targ_vals_dict.items():
                print(f"{key} target: {value}")

            myPars, myShocks, state_sols, sim_lc = calibration.calib_all(myPars, myShocks, modify_shocks = modify_shocks, 
                                                                do_wH_calib = do_wH_calib, do_dpi_calib = do_dpi_calib, do_phi_H_calib = do_phi_H_calib, 
                                                                do_eps_gg_calib=do_eps_gg_calib, do_eps_bb_calib=do_eps_bb_calib,
                                                                **calib_targ_vals_dict)
        else: # otherwise use default argument targets
            myPars, myShocks, state_sols, sim_lc = calibration.calib_all(myPars, myShocks, modify_shocks = modify_shocks,
                                                               do_wH_calib = do_wH_calib, do_dpi_calib = do_dpi_calib, do_phi_H_calib = do_phi_H_calib, 
                                                               do_eps_gg_calib=do_eps_gg_calib, do_eps_bb_calib=do_eps_bb_calib)

        calib_model_vals_dict = {   'alpha': calibration.alpha_moment_giv_sims(myPars, sim_lc), 
                                    'w0_mu': calibration.w0_moments(myPars, myShocks)[0], 'w0_sigma': calibration.w0_moments(myPars, myShocks)[1],
                                    'w1': calibration.w1_moment(myPars, myShocks), 'w2': calibration.w2_moment(myPars, myShocks),
                                    'wH': calibration.wH_moment(myPars, myShocks), 'phi_H': calibration.phi_H_moment(myPars, sim_lc['lab']),
                                    'dpi_BB': calibration.dpi_BB_moment(myPars), 'dpi_GG': calibration.dpi_GG_moment(myPars),
                                    'eps_gg': calibration.eps_gg_moment(myPars, myShocks), 'eps_bb': calibration.eps_bb_moment(myPars, myShocks)}

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
               data_moms_folder_path = data_moms_folder_path, outpath = output_path)
    
    return [state_sols, sim_lc]

def output(myPars: Pars, state_sols: Dict[str, np.ndarray], sim_lc: Dict[str, np.ndarray], targ_moments: Dict[str, np.ndarray], 
           model_moments: Dict[str, np.ndarray], tex: bool, get_targets: bool, data_moms_folder_path: str, outpath: str = None)-> None:
    if outpath is None:
        outpath = myPars.path + 'output/tabs_fit_figs/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    io.print_params_to_csv(myPars, outpath)
    if tex:
        tables.print_exog_params_to_tex(myPars, outpath)
        tables.print_endog_params_to_tex(myPars, targ_moments, model_moments, outpath)
        tables.print_w0_calib_to_tex(myPars, targ_moments, model_moments, outpath)
        # calibration.print_H_trans_to_tex(myPars, path)
    if get_targets:
        lab_mom_path = data_moms_folder_path + 'labor_moments.csv'
        plot_aggregates.plot_lab_aggs_and_moms(myPars, sim_lc, data_moms_path=lab_mom_path, outpath = outpath)
        emp_mom_path = data_moms_folder_path + 'emp_rate_moments.csv'
        plot_aggregates.plot_emp_aggs_and_moms(myPars, sim_lc, data_moms_path=emp_mom_path, outpath = outpath)
        wage_mom_path = data_moms_folder_path + 'wage_moments.csv'
        plot_aggregates.plot_wage_aggs_and_moms(myPars, data_moms_path=wage_mom_path, outpath = outpath)
        earn_mom_path = data_moms_folder_path + 'earnings_moments.csv'
        plot_aggregates.plot_earnings_aggs_and_moms(myPars, sim_lc, data_moms_path=earn_mom_path, outpath = outpath)
    plot_lc.plot_lc_profiles(myPars, sim_lc, outpath)

# run if main function
if __name__ == "__main__":
  pass 