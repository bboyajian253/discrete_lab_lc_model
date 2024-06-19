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

# My code
import my_toolbox as tb
import solver
import old_simulate as simulate
import calibrate as calib
import plot_lc as plot_lc
from pars_shocks_and_wages import Pars, Shocks

# Run the model

# Print the parameters
# Output the results and the associated graphs