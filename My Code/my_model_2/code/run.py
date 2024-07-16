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

# My code
import my_toolbox as tb
import solver
import simulate as simulate
import calibration
import plot_lc as plot_lc
from pars_shocks_and_wages import Pars, Shocks
import plot_moments 

# Run the model
def run_model(myPars: Pars, myShocks: Shocks, solve: bool = True, calib : bool = True, get_moments: bool = True, sim_no_calib  : bool = False, 
              output_flag: bool = True, no_tex: bool = False)-> List[Dict[str, np.ndarray]]:
    """
    Given the model parameters, solve, calibrate, simulate, and, if desired, output the results.
    (i) Solve the model
    (ii) Simulate the model
    (iii) Calibrate the model
    (iv) Output the results and model aggregates
    """
    
    #If solve, solve the model
    if solve:
        start_time = time.perf_counter()
        state_sols = solver.solve_lc(myPars)
        for label, value in state_sols.items():
            np.save(myPars.path + 'output/' + label + '_lc', value)
        tb.print_exec_time("Solver ran in", start_time)
    
    #always load state specific solutions
    sol_labels = ['c', 'lab', 'a_prime']
    state_sols = {}
    for label in sol_labels:
        state_sols[label] = np.load(myPars.path + 'output/' + label + '_lc.npy')

    #if no_calibrate_but_sim, simulate without calibrating
    if sim_no_calib:
        start_time = time.perf_counter()
        sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
        for label in sim_lc.keys():
            np.save(myPars.path + 'output/' + f'sim{label}.npy', sim_lc[label])
        tb.print_exec_time("Simulate ran in", start_time)
    elif calib:
        start_time = time.perf_counter()
        max_iters = myPars.max_iters
        #alpha, mean_labor, state_sols, sim_lc = calibration.calib_alpha(myPars, main_path, max_iters, lab_tol, lab_targ)
        if get_moments:
            alpha_lab_targ = calibration.get_alpha_targ(myPars)
            w0_mean_targ, w0_sd_targ = calibration.get_w0_mean_targ(myPars), calibration.get_w0_sd_targ(myPars)
            w1_targ = calibration.get_w1_targ(myPars)
            # w1_targ = 0.2
            w2_targ = calibration.get_w2_targ(myPars)
            # w2_targ = 0.2
        else:
            alpha_lab_targ, w0_mean_targ, w0_sd_targ, w1_targ, w2_targ = 0.40, 20.0, 5.0, 0.2, 0.2

        print("alpha_lab_targ", alpha_lab_targ)
        print("w0_mean_targ", w0_mean_targ)
        print("w0_sd_targ", w0_sd_targ)
        print("w1_targ", w1_targ)
        print("w2_targ", w2_targ)

        # calib_path = myPars.path + 'calibration/'
        calib_path = None
        calib_alpha, w0_weights, calib_w1, calib_w2, state_sols, sim_lc = calibration.calib_all(myPars, calib_path,  
                                                                                    alpha_lab_targ, w0_mean_targ, w0_sd_targ, 
                                                                                    w1_targ, w2_targ)
        calib_targ_vals_dict = { 'alpha': alpha_lab_targ, 'w0_mean': w0_mean_targ, 'w0_sd': w0_sd_targ, 'w1': w1_targ, 'w2': w2_targ}
        calib_model_vals_dict = {   'alpha': calibration.alpha_moment_giv_sims(myPars, sim_lc), 
                                    'w0_mean': calibration.w0_moments(myPars)[0], 'w0_sd': calibration.w0_moments(myPars)[1],
                                    'w1': calibration.w1_moment(myPars), 'w2': calibration.w2_moment(myPars) 
                                 }
        for label in sim_lc.keys():
            np.save(myPars.path + 'output/' + f'sim{label}.npy', sim_lc[label])
        tb.print_exec_time("Calibration ran in", start_time)

    #always load simulated life cycles
    sim_labels = ['c', 'lab', 'a', 'wage', 'lab_income']
    sim_lc = {}
    for label in sim_labels:
        sim_lc[label] = np.load(myPars.path + 'output/' + f'sim{label}.npy')

    #if output, output the results
    if output_flag:
        output(myPars, state_sols, sim_lc, calib_targ_vals_dict, calib_model_vals_dict, no_tex, get_moments)
    
    return [state_sols, sim_lc]

def output(myPars: Pars, state_sols: Dict[str, np.ndarray], sim_lc: Dict[str, np.ndarray], targ_moments: Dict[str, np.ndarray], model_moments: Dict[str, np.ndarray],no_tex, get_moments)-> None:
    # Print parameters
    calibration.print_params_to_csv(myPars)
    #calib_path = myPars.path + 'calibration/'
    if not no_tex:
        calibration.print_exog_params_to_tex(myPars)
        calibration.print_endog_params_to_tex(myPars, targ_moments, model_moments)
        calibration.print_wo_calib_to_tex(myPars, targ_moments, model_moments)
    if get_moments:
        plot_moments.plot_lab_aggs_and_moms(myPars, sim_lc)
        plot_moments.plot_wage_aggs_and_moms(myPars)
    plot_lc.plot_lc_profiles(myPars, sim_lc)

#Make run if main function
if __name__ == "__main__":
   
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/"

    my_lab_FE_grid = np.array([10.0, 15.0, 20.0])
    # my_lab_FE_grid = np.array([5.0, 10.0, 15.0])
    lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
    quad_wage_coeffs = [-0.000, -0.02, -0.02, -0.02] 
    cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

    num_FE_types = len(my_lab_FE_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])
    
    w_coeff_grid[0, :] = [my_lab_FE_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
    w_coeff_grid[1, :] = [my_lab_FE_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
    w_coeff_grid[2, :] = [my_lab_FE_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
    # w_coeff_grid[3, :] = [my_lab_FE_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

    print("intial wage coeff grid")
    print(w_coeff_grid)

    my_lab_FE_weights = tb.gen_even_weights(w_coeff_grid)
    print("even wage coeff grid")
    print(my_lab_FE_weights)


    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -500.0, a_max = 500.0, H_grid=np.array([0.0, 1.0]),
                nu_grid_size=1, alpha = 0.45, sim_draws=1000, lab_FE_grid = my_lab_FE_grid, lab_FE_weights = my_lab_FE_weights,
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 1, sigma_util = 0.9999,
                print_screen=0)
    # Set up the shocks
    myShocks = Shocks(myPars)
    myPars.path = main_path
    sols, sims =run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, get_moments = True, output_flag = True, no_tex = False)
