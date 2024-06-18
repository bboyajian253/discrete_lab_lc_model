"""
calibration.py

This file contains the calibration code for the model.

Author: Ben
Date: 2024-06-17 15:01:32
"""

# Import packages
import numpy as np

import my_toolbox as tb
import solver
import pars_shocks_and_wages as ps
import old_simulate as simulate
import plot_lc as plot_lc


def calib_alpha(myPars : ps.Pars, main_path : str) -> float:
    start_alpha = myPars.alpha
    alpha_iters = 100
    lab_tol = 0.1
    lab_targ = 0.40
    # solve model for a given alpha
    shocks = ps.Shocks(myPars)
    state_sols = solver.solve_lc(myPars)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    plot_lc.plot_lc_profiles(myPars, sim_lc)
    
    labor_sims = sim_lc['c']
    print(labor_sims)
    # get the mean labor worked across all labor fixed effect groups
    
    mean_lab = np.mean(labor_sims)
    print(mean_lab)

    
    # check if model matches mean labor worked for one labor fixed effect group = 40
    # start with the low ability group and a guees for alpha of 0.45
    # write parameters to a file
    # if it is within a certain tolerance then return the alpha
    # if not then adjust alpha and repeat
    pass

#put a run if main function here
if __name__ == "__main__":
        main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/calibration/"
        myPars = ps.Pars(main_path, J=50, a_grid_size=100, a_min= -500.0, a_max = 500.0, 
                    H_grid=np.array([1.0]), nu_grid_size=1, alpha = 0.45, sim_draws=1000,
                    print_screen=3)
        
        calib_alpha(myPars, main_path)
