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

# Run the model
def run_model(myPars: Pars, myShocks: Shocks, solve: bool = True, calib : bool = True, sim_no_calib  : bool = False, output_flag: bool = True, no_tex: bool = False)-> List[Dict[str, np.ndarray]]:
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
            np.save(myPars.path + label + '_lc', value)
        tb.print_exec_time("Solver ran in", start_time)
    
    #always load state specific solutions
    sol_labels = ['c', 'lab', 'a_prime']
    state_sols = {}
    for label in sol_labels:
        state_sols[label] = np.load(myPars.path + label + '_lc.npy')

    #if no_calibrate_but_sim, simulate without calibrating
    if sim_no_calib:
        start_time = time.perf_counter()
        sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
        for label in sim_lc.keys():
            np.save(myPars.path + f'sim{label}.npy', sim_lc[label])
        tb.print_exec_time("Simulate ran in", start_time)
    elif calib:
        start_time = time.perf_counter()
        max_iters = myPars.max_iters
        #alpha, mean_labor, state_sols, sim_lc = calibration.calib_alpha(myPars, main_path, max_iters, lab_tol, lab_targ)
        lab_targ, w1_targ, w2_targ = 0.4, 0.2, 0.2
        calib_path = myPars.path + 'calibration/'
        calib_alpha, calib_w1, calib_w2, state_sols, sim_lc = calibration.calib_all(myPars, calib_path, max_iters, 
                                                                                    lab_targ, w1_targ, w2_targ)
        for label in sim_lc.keys():
            np.save(myPars.path + f'sim{label}.npy', sim_lc[label])
        tb.print_exec_time("Calibration ran in", start_time)

    #always load simulated life cycles
    sim_labels = ['c', 'lab', 'a', 'wage', 'lab_income']
    sim_lc = {}
    for label in sim_labels:
        sim_lc[label] = np.load(myPars.path + f'sim{label}.npy')

    #if output, output the results
    if output_flag:
        output(myPars, state_sols, sim_lc, no_tex)
    
    return [state_sols, sim_lc]

def output(myPars: Pars, state_sols: Dict[str, np.ndarray], sim_lc: Dict[str, np.ndarray], no_tex)-> None:
    # Print parameters
    calibration.print_params_to_csv(myPars)
    #calib_path = myPars.path + 'calibration/'
    if not no_tex:
        calibration.print_exog_params_to_tex(myPars)
        calibration.print_endog_params_to_tex(myPars)
        calibration.print_wage_coeffs_to_tex(myPars)
    # Output the results and the associated graphs

    plot_lc.plot_lc_profiles(myPars, sim_lc)

#Make run if main function
if __name__ == "__main__":
   
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/"

    # my_lab_FE_grid = np.array([10.0, 20.0, 30.0, 40.0])
    my_lab_FE_grid = np.array([10.0, 10.0])
    lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
    quad_wage_coeffs = [-0.000, -0.02, -0.02, -0.02] 
    cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

    num_FE_types = len(my_lab_FE_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])
    
    w_coeff_grid[0, :] = [my_lab_FE_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
    w_coeff_grid[1, :] = [my_lab_FE_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
    # w_coeff_grid[2, :] = [my_lab_FE_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
    # w_coeff_grid[3, :] = [my_lab_FE_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

    print("intial wage coeff grid")
    print(w_coeff_grid)

    my_lab_FE_weights = tb.gen_even_weights(w_coeff_grid)
    print("even wage coeff grid")
    print(my_lab_FE_weights)


    myPars = Pars(main_path, J=60, a_grid_size=501, a_min= -500.0, a_max = 500.0, H_grid=np.array([0.0, 1.0]),
                nu_grid_size=1, alpha = 0.45, sim_draws=1000, lab_FE_grid = my_lab_FE_grid, lab_FE_weights = my_lab_FE_weights,
                wage_coeff_grid = w_coeff_grid, max_iters = 100, sigma_util = 0.9999,
                print_screen=0)
    # Set up the shocks
    myShocks = Shocks(myPars)
    # Run the model no calibration
    # run_model(myPars, myShocks, solve = True, calib = False, sim_no_calib = True, output_flag = True)
    # Run the model with calibration
    # myPars.path = myPars.path + 'test_calib/'
    # run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, output_flag = True)
    myPars.path = main_path
    run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, output_flag = True, no_tex = True)
