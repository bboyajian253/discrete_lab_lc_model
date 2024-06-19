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
import old_simulate as simulate
import calibration as calib
import plot_lc as plot_lc
from pars_shocks_and_wages import Pars, Shocks

# Run the model
def run_model(myPars: Pars, myShocks: Shocks)-> List[Dict[str, float]]:
    #If solve, solve the model
    #always load state specific solutions
    #if no_calibrate_but_sim, simulate without calibrating
    #always load simulated life cycles
    #if output, output the results

def output(myPars: Pars, state_sols: Dict[str, np.ndarray], sim_lc: Dict[str, np.ndarray])-> None:
    # Print parameters
    # Output the results and the associated graphs
    pass