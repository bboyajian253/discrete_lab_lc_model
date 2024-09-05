"""
Created on 2024-05-18 00:37:20

@author: Ben Boyaian
"""
import my_toolbox
import my_toy_ls_model
import solver
import simulate
import pars_shocks_and_wages 
import numpy as np
from time import time
import plot_lc

def solver_test_main1() :
    myPars = pars_shocks_and_wages.Pars(print_screen=0, a_grid_size=100)
    my_toolbox.print_exec_time("Pars 1 initialized in", start_time)

    new_start_time = time()
    solver.solve_lc(myPars)
    my_toolbox.print_exec_time("Solver 1 for pars 1 compiled and ran after", new_start_time) 

    new_start_time = time()
    solver.solve_lc(myPars)
    my_toolbox.print_exec_time("Solver 2 for pars 1 ran (since already compiled) after", new_start_time)

    new_start_time = time()
    myPars = pars_shocks_and_wages.Pars(print_screen=0, a_grid_size=300)
    my_toolbox.print_exec_time("Pars 2 initialized in", new_start_time)

    new_start_time = time()
    solver.solve_lc(myPars)
    my_toolbox.print_exec_time("Solver 1 for pars 2 ran after", new_start_time) 

    new_start_time = time()
    solver.solve_lc(myPars)
    my_toolbox.print_exec_time("Solver 2 for pars 2 ran (since already compiled) after", new_start_time)

def simulate_test(start_time):

    my_toolbox.print_exec_time("Beginning to initilize pars", start_time)

    new_start_time = time()
    myPars = pars_shocks_and_wages.Pars(print_screen=0, a_grid_size=300, lab_fe_grid=np.array([0.5, 1.0, 1.5, 2.0]))
    my_toolbox.print_exec_time("Pars 1 compiled and initialized in", new_start_time)

    new_start_time = time()
    myShocks = pars_shocks_and_wages.Shocks(myPars)
    my_toolbox.print_exec_time("Shocks 1 initialized in", new_start_time)

    new_start_time = time()
    state_sols = solver.solve_lc(myPars)
    my_toolbox.print_exec_time("Solver 1 for pars 1 ran after", new_start_time) 

    new_start_time = time()
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    my_toolbox.print_exec_time("Simulate 1 for pars 1 ran after", new_start_time)

    new_start_time = time()
    myPath = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_1/output/"
    plot_lc.plot_lc_profiles(myPars, sim_lc, myPath)
    my_toolbox.print_exec_time("Plot 1 for pars 1 ran after", new_start_time)

start_time = time()
print("Running main")

simulate_test(start_time)

my_toolbox.print_exec_time("Main.py executed in", start_time) 
