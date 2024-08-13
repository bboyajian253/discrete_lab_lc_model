"""
My Model 2 - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
# Import packages
# General
import time
import numpy as np
# My code
import pars_shocks as ps
from pars_shocks import Pars, Shocks
import model_uncert as model
import my_toolbox as tb
import solver
import simulate as simulate
import plot_lc as plot_lc
import run 

def main_1():
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

    my_lab_FE_grid = np.array([5.0, 10.0, 15.0, 20.0])
    # my_lab_FE_grid = np.array([5.0, 10.0, 15.0])
    my_lab_FE_grid = np.log(my_lab_FE_grid)
    lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
    quad_wage_coeffs = [-0.000, -0.02, -0.02, -0.02] 
    cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

    num_FE_types = len(my_lab_FE_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])
    
    w_coeff_grid[0, :] = [my_lab_FE_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
    w_coeff_grid[1, :] = [my_lab_FE_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
    w_coeff_grid[2, :] = [my_lab_FE_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
    w_coeff_grid[3, :] = [my_lab_FE_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

    print("intial wage coeff grid")
    print(w_coeff_grid)

    my_lab_FE_weights = tb.gen_even_weights(w_coeff_grid)

    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), H_weights=np.array([0.5, 0.5]),
                nu_grid_size=1, alpha = 0.45, sim_draws=1000, lab_FE_grid = my_lab_FE_grid, lab_FE_weights = my_lab_FE_weights,
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 100, sigma_util = 0.9999,
                print_screen=0)
    myShocks = Shocks(myPars)
    myPars.path = main_path
    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, 
                          get_moments = True, output_flag = True, tex = True)
    

#run stuff here
start_time = time.perf_counter()
print("Running main")
main_1()
tb.print_exec_time("Main.py executed in", start_time) 
