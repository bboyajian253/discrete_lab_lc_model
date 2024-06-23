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
import calibration as calib
import plot_lc as plot_lc
from pars_shocks_and_wages import Pars, Shocks

# Run the model
def run_model(myPars: Pars, myShocks: Shocks, solve: bool = True, calib : bool = True, sim_no_calib  : bool = False, output_flag: bool = True)-> List[Dict[str, np.ndarray]]:
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

    #always load simulated life cycles
    sim_labels = ['c', 'lab', 'a', 'wage', 'lab_income']
    sim_lc = {}
    for label in sim_labels:
        sim_lc[label] = np.load(myPars.path + f'sim{label}.npy')

    #if output, output the results
    if output_flag:
        output(myPars, state_sols, sim_lc)
    
    return [state_sols, sim_lc]

def output(myPars: Pars, state_sols: Dict[str, np.ndarray], sim_lc: Dict[str, np.ndarray])-> None:
    # Print parameters
    calib.print_params_to_csv(myPars)
    # Output the results and the associated graphs
    plot_lc.plot_lc_profiles(myPars, sim_lc)

#Make run if main function
if __name__ == "__main__":
   
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/"
    # Set up the parameters
    FE_grid = np.array([1.0, 2.0, 3.0])
    num_FE_types = len(FE_grid)

    w_coeff_grid = np.zeros([num_FE_types, 4])
    # for i in range(num_FE_types):
    #     w_coeff_grid[i, :] = [10.0*(i+1), 0.5*(i), -0.010*(i), 0.0]
    w_coeff_grid[0, :] = [10.0, 0.0, -0.00, 0.0]
    w_coeff_grid[1, :] = [20.0, .5, -0.010, 0.0]
    w_coeff_grid[2, :] = [30.0, 1.0, -0.020, 0.0]

    #should alpha be 0.45 or 0.70? Seems like 0.45 is the correct value will calibrate
    myPars = Pars(main_path, J=50, a_grid_size=100, a_min= -300.0, a_max = 300.0, sigma_util=3.0,
                     lab_FE_grid=FE_grid, H_grid=np.array([1.0]), nu_grid_size=1, alpha = 0.45, sim_draws=1000, 
                     print_screen=3)
 
    # Set up the shocks
    myShocks = Shocks(myPars)
    # Run the model
    run_model(myPars, myShocks, solve = True, calib = False, sim_no_calib = True, output_flag = True)