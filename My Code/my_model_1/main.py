"""
Created on 2024-05-18 00:37:20

@author: Ben Boyaian
"""
import my_toolbox
import my_toy_ls_model
import solver
import pars_shocks_and_wages 
import numpy as np
from time import time


start_time = time()
print("Running main")

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


my_toolbox.print_exec_time("Main.py executed in", start_time) 




