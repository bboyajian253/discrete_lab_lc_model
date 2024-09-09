"""
@Author: Ben Boyajian
@Date: 2024-09-06 12:01:17
File: plot_inquality.py
project: model_uncert
"""

# imports
# general
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Tuple, Dict

# my code
from pars_shocks import Pars

def plot_var_log_sim(myPars: Pars, sim: np.ndarray, y_axis_lab: str, out_path: str, quietly = False) -> Tuple[Figure, Axes]:
    """
    plot the variance of the log of the simulated variable by age
    """
    log_sim = np.log(sim, where = sim > 0)
    var_log_sim_by_age = weighted_var_sim_by_age(myPars, log_sim)
    fig, ax = plot_lc_mom_by_age(var_log_sim_by_age, myPars.age_grid, out_path, y_axis_lab, quietly
    return fig, ax