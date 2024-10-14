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
import model_uncert as model
import my_toolbox as tb

def plot_var_log_sim(myPars: Pars, sim: np.ndarray, y_axis_lab: str, outpath: str = None, full_age_grid: bool = False, quietly: bool = False) -> Tuple[Figure, Axes]:
    """
    plot the variance of the log of the simulated variable by age
    """
    log_sim = np.log(sim, where = sim > 0)
    var_log_sim_by_age = weighted_var_sim_by_age(myPars, log_sim)
    my_age_grid = myPars.age_grid[:31] # only plot up to age 55
    # my_age_grid = myPars.age_grid[:41] # only plot up to age 65
    if full_age_grid:
        my_age_grid = myPars.age_grid
    fig, ax = tb.plot_lc_mom_by_age(var_log_sim_by_age, my_age_grid, y_axis_lab, quietly = quietly)
    return fig, ax

def weighted_var_sim_by_age(myPars: Pars, sim: np.ndarray) -> np.ndarray:
    """
     calculate the variance of the simulated variable by age and return it as a 1D array
    """
    weighted_sim_by_age = model.gen_weighted_sim(myPars, sim)
    weighted_mean_by_age = np.sum(weighted_sim_by_age, axis = tuple(range(sim.ndim - 1))) # sum over all dimensions except the last one
    dev_from_mean = sim - weighted_mean_by_age
    squared_dev = dev_from_mean**2
    weighted_var_by_age = np.sum(model.gen_weighted_sim(myPars, squared_dev), axis = tuple(range(sim.ndim - 1)))
    return weighted_var_by_age
    # var_by_age = np.var(sim, axis = tuple(range(sim.ndim - 1)))
    # return var_by_age

def wperc_sim_by_age(myPars: Pars, sim: np.ndarray, perc:float) -> np.ndarray:
    """
    calculate the weighted percentile of the simulated variable by age and return it as a 1D array
    """
    weight0 = myPars.lab_fe_weights
    weight1 = myPars.H_type_perm_weights
    weight2 = np.ones(myPars.sim_draws)

    #reshape to make the weights broadcastable
    weight0_reshape = weight0[:, np.newaxis, np.newaxis]
    weight1_reshape = weight1[np.newaxis, :, np.newaxis]
    weight2_reshape = weight2[np.newaxis, np.newaxis, :]

    #combine the weights
    combined_weights = weight0_reshape * weight1_reshape * weight2_reshape
    perc_sim_by_age = tb.collapse_to_last_dim_wperc(sim, combined_weights, perc)
    return perc_sim_by_age

def wperc_log_lab_earn_by_age(myPars: Pars, sims: Dict[str, np.ndarray], percentile:float) -> np.ndarray:
    """
    calculate the weighted percentile of the log of labor earnings by age and return it as a 1D array
    """
    log_lab_earn = np.log(sims['lab_earnings'], where = sims['lab_earnings'] > 0)
    perc_log_lab_earn_by_age = wperc_sim_by_age(myPars, log_lab_earn, percentile)
    return perc_log_lab_earn_by_age

def plot_many_sim_perc_ratio(myPars: Pars, sim: np.ndarray, y_axis_label_root: str, outpath: str, quietly: bool = False
                             ) -> Tuple[Figure, Axes]:
    """
    plots the 5th, 10th, 50th, and 90th percentiles of the simulated variable and their ratios by age
    returns
    """
    sim_age_90p = wperc_sim_by_age(myPars, sim, 90)
    sim_age_50p = wperc_sim_by_age(myPars, sim, 50)
    sim_age_10p = wperc_sim_by_age(myPars, sim, 10)
    sim_age_5p = wperc_sim_by_age(myPars, sim, 5)
    sim_age_90_10p = sim_age_90p / sim_age_10p
    sim_age_90_50p = sim_age_90p / sim_age_50p
    sim_age_50_10p = sim_age_50p / sim_age_10p
    sim_age_90_5p = sim_age_90p / sim_age_5p
    sim_age_50_5p = sim_age_50p / sim_age_5p
    my_age_grid = myPars.age_grid
    my_age_grid = myPars.age_grid[:31] # only plot up to age 55

    fig_90, ax_90 = tb.plot_lc_mom_by_age(sim_age_90p, my_age_grid,  "90th Percentile of " + y_axis_label_root, quietly = quietly)
    fig_50, ax_50 = tb.plot_lc_mom_by_age(sim_age_50p, my_age_grid,  "50th Percentile of " + y_axis_label_root, quietly = quietly)
    fig_10, ax_10 = tb.plot_lc_mom_by_age(sim_age_10p, my_age_grid,  "10th Percentile of " + y_axis_label_root, quietly = quietly)
    fig_5, ax_5 = tb.plot_lc_mom_by_age(sim_age_5p, my_age_grid,  "5th Percentile of " + y_axis_label_root, quietly = quietly)
    fig_90_10, ax_90_10 = tb.plot_lc_mom_by_age(sim_age_90_10p, my_age_grid,  "90th/10th Percentile Ratio of " + y_axis_label_root, quietly = quietly)
    fig_90_50, ax_90_50 = tb.plot_lc_mom_by_age(sim_age_90_50p, my_age_grid,  "90th/50th Percentile Ratio of " + y_axis_label_root, quietly = quietly)
    fig_50_10, ax_50_10 = tb.plot_lc_mom_by_age(sim_age_50_10p, my_age_grid,  "50th/10th Percentile Ratio of " + y_axis_label_root, quietly = quietly)
    fig_90_5, ax_90_5 = tb.plot_lc_mom_by_age(sim_age_90_5p, my_age_grid, "90th/5th Percentile Ratio of " + y_axis_label_root, quietly = quietly)
    fig_50_5, ax_50_5 = tb.plot_lc_mom_by_age(sim_age_50_5p, my_age_grid, "50th/5th Percentile Ratio of " + y_axis_label_root, quietly = quietly)

    return[(fig_90, ax_90), (fig_50, ax_50), (fig_10, ax_10), (fig_5, ax_5), 
           (fig_90_10, ax_90_10), (fig_90_50, ax_90_50), (fig_50_10, ax_50_10), 
           (fig_90_5, ax_90_5), (fig_50_5, ax_50_5)]
    # fig_list =[fig_90, fig_50, fig_10, fig_5, fig_90_10, fig_90_50, fig_50_10, fig_90_5, fig_50_5]
    # ax_list = [ax_90, ax_50, ax_10, ax_5, ax_90_10, ax_90_50, ax_50_10, ax_90_5, ax_50_5]
    # return fig_list, ax_list


    