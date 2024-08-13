"""
plot_lc.py

Created on 2024-05-29 11:30:48
Created by Ben Boyaian

plots life-cycle simulations of calibrated model 

"""
# import packages
# General
import numpy as np
import matplotlib.pyplot as plt
import my_toolbox
from typing import List, Dict
import csv
# My code
from pars_shocks import Pars


def plot_lc_profiles(myPars : Pars, sim_lc: Dict[str, np.ndarray], path: str = None)-> None:
    
    #define path
    if path is None:
        path = myPars.path + 'output/'
    #Generate variable labels lists
    var_lables = ['Consumption', 'Labor', 'Assets', 'Wage', 'Labor Income']
    #Generate the short names for the variables i think this should be their keys in the sim_lc dictionary
    var_names_short = ['c', 'lab', 'a', 'wage', 'lab_income']
    #for each variable in the variable list zip the short names and label names together and loop through them
    for label, short_name in zip(var_lables, var_names_short):
        #initialize the life-cycle shells
        #this will change as i introduce retirement
        # some lasts depend on or are equal to the retirment age 
        j_last = myPars.J

        age_grid = myPars.age_grid[:j_last]

        #prep the sim variable for estimation
        if short_name == 'a':
            values = sim_lc[short_name][:, :, :, :, :j_last+1]
        else:
            values = sim_lc[short_name][:, :, :, :, :j_last] # for each var get the the array of choices until the last age
        log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number

        #Plot life-cycle profiles
        for modifier in ['', 'log']:
            if myPars.print_screen >= 2:
                print(modifier, label)
            fig, ax = plt.subplots()

            if modifier == 'log':
                lc = log_values
            else:
                lc = values

            #iterate through labor fixed effect groups (this is basically ability groups)
            for h_ind in range(myPars.H_grid_size):
                for lab_fe_ind in range(myPars.lab_FE_grid_size):    
                    #get the mean of the values over the labor fixed effects and health types 
                    lc_mean = np.average(lc[lab_fe_ind, h_ind, 0, :], axis=(0))
                    myLab = f"FE:{round(np.exp(myPars.lab_FE_grid[lab_fe_ind]))} H:{round(myPars.H_grid[h_ind])}"
                    if short_name == 'a':
                        a_age_grid = np.append(age_grid, age_grid[-1] + 1)
                        ax.plot(a_age_grid, lc_mean, label=myLab) 
                    else:
                        ax.plot(age_grid, lc_mean, label=myLab)
            
            #specify axes and legend
            ax.set_xlabel('Age')
            ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2]) #set the x axis limits
            ax.set_ylabel(modifier + ' ' + label)
            # if short_name == 'lab' and modifier != 'log':
                # ax.set_ylim([0, 1])
            # elif short_name == 'a' and modifier != 'log':
            #     ax.set_ylim([myPars.a_min - 2, myPars.a_max + 2])

            ax.legend()

            #save the figure
            fullpath = path + f'fig_lc_{short_name}_{modifier}.pdf'
            fig.savefig(fullpath, bbox_inches='tight')
            plt.close()

            #save the data
            fullpath =  path + f'fig_lc_{short_name}_{modifier}.csv'
            with open(fullpath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['age'] + list(age_grid))
                for row in lc:
                    writer.writerows(['model'] + list(lc))

# probably should plot policy functions as a function of states evetually
def plot_c_by_a(myPars : Pars, sim_lc):
    pass
    # fig, ax = plt.subplots()
    # for lab_FE_ind in range(myPars.lab_FE_grid_size):
    #     c_mean = np.average(sim_lc['c'][lab_FE_ind, 0, 0, :, :], axis=(0))
    #     ax.plot(myPars.a_grid, label = f'Age: {age}')

    
    
    
    
        
  


if __name__ == "__main__":
    pass