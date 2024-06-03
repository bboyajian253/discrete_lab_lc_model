"""
My Model 2 - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
import pars_shocks_and_wages as ps
import model_no_uncert as model
import my_toolbox as tb
import solver
import old_simulate as simulate
import old_plot_lc as plot_lc
import numpy as np

main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/"

def main():
    print("Running main")

    myPars = ps.Pars(main_path, J=25, a_grid_size=5, a_min= -5.0, a_max = 5.0, lab_FE_grid=np.array([0.0, 0.5, 1.0]), H_grid=np.array([0.0]), nu_grid_size=1, alpha = 0.5, )

    myShocks = ps.Shocks(myPars)
    print("Path:", myPars.path)
    sols = solver.solve_lc(myPars)
    sim_lc = simulate.sim_lc(myPars, myShocks, sols)
    #print("Simulated LC:", sim_lc)
    plot_lc.plot_lc_profiles(myPars, sim_lc)

#run stuff here
main()