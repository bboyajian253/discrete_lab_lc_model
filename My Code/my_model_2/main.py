"""
My Model 2 - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
import time
import pars_shocks_and_wages as ps
import model_no_uncert as model
import my_toolbox as tb
import solver
import old_simulate as simulate
import plot_lc as plot_lc
import numpy as np
 

def simulate_test(start_time, main_path):

    tb.print_exec_time("Beginning to initilize pars", start_time)

    new_start_time = time.perf_counter()
    #fe_grid = np.arange(6.0)
    FE_grid = np.array([2.30259,2.99573,3.4012]) #calibrated so that wages = 10, 20, 30 respectively
    num_FE_types = len(FE_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])
    w_coeff_grid[0, :] = [10.0, 0.0, 0.0, 0.0]
    w_coeff_grid[1, :] = [20.0, 1.0, 0.0, 0.0]
    w_coeff_grid[2, :] = [30.0, 2.0, 0.0, 0.0]
    #should alpha be 0.45 or 0.70?
    myPars = ps.Pars(main_path, J=10, a_grid_size=100, a_min= -30.0, a_max = 30.0, 
                     lab_FE_grid=FE_grid, H_grid=np.array([0.0]), nu_grid_size=1, alpha = 0.45, sim_draws=1000, 
                     wage_coeff_grid=w_coeff_grid,
                     print_screen=3)
    
    tb.print_exec_time("Pars 1 compiled and initialized in", new_start_time)

    new_start_time = time.perf_counter()
    myShocks = ps.Shocks(myPars)
    tb.print_exec_time("Shocks 1 initialized in", new_start_time)

    new_start_time = time.perf_counter()
    state_sols = solver.solve_lc(myPars)
    tb.print_exec_time("Solver 1 for pars 1 ran after", new_start_time) 

    new_start_time = time.perf_counter()
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    tb.print_exec_time("Simulate 1 for pars 1 ran after", new_start_time)

    new_start_time = time.perf_counter()
    plot_lc.plot_lc_profiles(myPars, sim_lc)
    tb.print_exec_time("Plot 1 for pars 1 ran after", new_start_time)


def main_1(main_path):
    
    fe_grid = np.arange(6.0)
    myPars = ps.Pars(main_path, J=50, a_grid_size=300, a_min= -5.0, a_max = 5.0, 
                     lab_FE_grid=fe_grid, H_grid=np.array([0.0,1.0]), nu_grid_size=1, alpha = 0.5, sim_draws=1000,)

    myShocks = ps.Shocks(myPars)
    print("Path:", myPars.path)
    sols = solver.solve_lc(myPars)
    sim_lc = simulate.sim_lc(myPars, myShocks, sols)
    #print("Simulated LC:", sim_lc)
    plot_lc.plot_lc_profiles(myPars, sim_lc)

#run stuff here
start_time = time.perf_counter()
print("Running main")
main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/"

simulate_test(start_time, main_path)

tb.print_exec_time("Main.py executed in", start_time) 
