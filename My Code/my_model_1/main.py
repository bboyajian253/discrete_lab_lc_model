"""
Created on 2024-05-18 00:37:20

@author: Ben Boyaian
"""
import my_toolbox
import my_toy_ls_model
import solver
import pars_and_shocks 
import numpy as np
from time import time


start_time = time()
print("Running main")

myPars = pars_and_shocks.Pars(print_screen=0)
my_toolbox.print_exec_time("Pars initialized in", start_time)

new_start_time = time()
solver.solve_lc(myPars)
my_toolbox.print_exec_time("Solver 1 compiled and ran after", new_start_time) 

new_start_time = time()
solver.solve_lc(myPars)
my_toolbox.print_exec_time("Solver 2 ran (since already compiled) after", new_start_time)

my_toolbox.print_exec_time("Main.py executed in", start_time) 




