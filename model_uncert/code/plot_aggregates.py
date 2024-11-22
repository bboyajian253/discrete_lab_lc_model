"""
plot_aggregates.py

Created on 2024-07-09 01:11:05
Created by Ben Boyaian

plots simulated moments and matched moments together to compare fit 

"""
#import stuff
#General
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.axes import Axes
from typing import  List, Dict, Tuple
import csv
import time
import os

# My code
from pars_shocks import Pars, Shocks
import model_uncert as model
import my_toolbox as tb
import solver
import simulate
import io_manager as io

def plot_wage_aggs_and_moms(myPars: Pars, data_moms_path: str = None, outpath: str = None)-> None:
    if outpath == None:
        outpath = myPars.path + 'output/'
    if data_moms_path == None:
        data_moms_path = myPars.path + '/input/wage_moments.csv'
    # calcualte weighted mean wages by age
    weighted_wages = model.gen_weighted_wage_hist(myPars, Shocks(myPars)) 
    mean_weighted_wages = np.sum(weighted_wages, axis=tuple(range(weighted_wages.ndim - 1)))
    log_wages = model.gen_wlog_wage_hist(myPars, Shocks(myPars))
    mean_wlog_wages = np.sum(log_wages, axis=tuple(range(log_wages.ndim - 1)))
    
    # plot that shit
    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = mean_weighted_wages[:j_last]
    log_values = mean_wlog_wages[:j_last]
    sim_key_label = "Simulated"
    data_moments_label = 'From the data'
    # log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number
    data_moments_col_ind = 1
    data_moments = tb.read_specific_column_from_csv(data_moms_path, data_moments_col_ind) # 1 means read the second column
    log_moments_col_ind = 3
    log_moments = tb.read_specific_column_from_csv(data_moms_path, log_moments_col_ind)
    for modifier in ['','log']:
        if modifier == 'log':
            sim_values = log_values
            mom_values = log_moments
            sim_y_label = "Average Log Wage (Weighted)"
        else:
            sim_values = values
            mom_values = data_moments
            sim_y_label = "Average Wage (Weighted)"
        if myPars.print_screen >= 2:
                print(sim_y_label)
        fig, ax = plt.subplots()
        ax.plot(age_grid, sim_values, label = sim_key_label)
        ax.plot(age_grid, mom_values, label = data_moments_label)
        # specify axis and labels
        ax.set_xlabel('Age')
        ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2]) 
        ax.set_ylabel(sim_y_label)
        ax.legend()

        short_name = 'wage'

        #save the figure
        fullpath = outpath + f'fig_fit_{short_name}_{modifier}.pdf'
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        #save the data
        fullpath =  outpath + f'fig_fit_{short_name}_{modifier}.csv'
        with open(fullpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['age'] + list(age_grid))
            writer.writerow(['model'] + list(sim_values))
            writer.writerow(['data'] + list(data_moments))

def weighted_avg_lab_by_age(myPars: Pars, sim_lc: Dict[str, np.ndarray])-> np.ndarray:
    labor_sims = sim_lc['lab'][:, :, :, :myPars.J]
    weighted_labor_sims = model.gen_weighted_sim(myPars, labor_sims) 
    mean_lab_by_age = np.sum(weighted_labor_sims, axis = tuple(range(weighted_labor_sims.ndim-1)))
    return mean_lab_by_age

def plot_earnings_aggs_and_moms(myPars: Pars, sim_lc: Dict[str, np.ndarray], data_moms_path: str = None, outpath: str = None)-> None:
    if outpath == None:
        outpath = myPars.path + 'output/'
    if data_moms_path == None:
        data_moms_path = myPars.path + '/input/earnings_moments.csv'
    # calcualte weighted mean earnings by age
    avg_earn_age = weighted_avg_earnings_by_age(myPars, sim_lc)
    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = avg_earn_age
    sim_key_label = "Simulated"
    data_moments_label = 'From the data'
    log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number
    data_moments_col_ind = 1
    data_moments = tb.read_specific_column_from_csv(data_moms_path, data_moments_col_ind) # 1 means read the second column
    log_moments = np.log(np.where(data_moments > 0, data_moments, 1e-3))
    for modifier in ['','log']:
        if modifier == 'log':
            sim_values = log_values
            mom_values = log_moments
            sim_y_label = "Average Log Earnings (Weighted)"
        else:
            sim_values = values
            mom_values = data_moments
            sim_y_label = "Average Earnings (Weighted)"
        if myPars.print_screen >= 2:
            print(sim_y_label)
        
        fig, ax = plt.subplots()
        ax.plot(age_grid, sim_values, label = sim_key_label)
        ax.plot(age_grid, mom_values, label = data_moments_label)
        # specify axis and labels
        ax.set_xlabel('Age')
        ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2])
        ax.set_ylabel(sim_y_label)
        ax.legend()

        short_name = 'earnings'

        #save the figure
        fullpath = outpath + f'fig_fit_{short_name}_{modifier}.pdf'
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        #save the data
        fullpath =  outpath + f'fig_fit_{short_name}_{modifier}.csv'
        with open(fullpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['age'] + list(age_grid))
            writer.writerow(['model'] + list(sim_values))
            writer.writerow(['data'] + list(data_moments))
    

def weighted_avg_earnings_by_age(myPars: Pars, sim_lc: Dict[str, np.ndarray])-> np.ndarray:
    earn_sims = sim_lc['lab_earnings'][:, :, :, :myPars.J]
    weighted_earn_sims = model.gen_weighted_sim(myPars, earn_sims)
    mean_earn_by_age = np.sum(weighted_earn_sims, axis = tuple(range(weighted_earn_sims.ndim-1)))
    # mean_earn_by_age = np.mean(earn_sims, axis = tuple(range(earn_sims.ndim-1)))
    return mean_earn_by_age

def plot_lab_aggs_and_moms(myPars: Pars, sim_lc: Dict[str, np.ndarray], data_moms_path: str = None, outpath: str = None)-> None:
    if outpath == None:
        outpath = myPars.path + 'output/'
    if data_moms_path == None: 
        data_moms_path = myPars.path + '/input/labor_moments.csv'
    avg_lab = weighted_avg_lab_by_age(myPars, sim_lc)
    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = avg_lab
    sim_key_label = "Simulated"
    data_moments_label = 'From the data'
    log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number
    data_moments_col_ind = 1
    data_moments = tb.read_specific_column_from_csv(data_moms_path, data_moments_col_ind) # 1 means read the second column
    log_moments = np.log(np.where(data_moments > 0, data_moments, 1e-3))
    for modifier in ['','log']:
        if modifier == 'log':
            sim_values = log_values
            mom_values = log_moments
            sim_y_label = "Average Log Labor (Weighted)"
        else:
            sim_values = values
            mom_values = data_moments
            sim_y_label = "Average Labor (Weighted)"
        if myPars.print_screen >= 2:
            print(sim_y_label)
        
        fig, ax = plt.subplots()
        ax.plot(age_grid, sim_values, label = sim_key_label)
        ax.plot(age_grid, mom_values, label = data_moments_label)
        # specify axis and labels
        ax.set_xlabel('Age')
        ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2])
        ax.set_ylabel(sim_y_label)
        ax.legend()

        short_name = 'lab'

        #save the figure
        fullpath = outpath + f'fig_fit_{short_name}_{modifier}.pdf'
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        #save the data
        fullpath =  outpath + f'fig_fit_{short_name}_{modifier}.csv'
        with open(fullpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['age'] + list(age_grid))
            writer.writerow(['model'] + list(sim_values))
            writer.writerow(['data'] + list(data_moments))
            
def weighted_emp_rate_by_age(myPars: Pars, sim_lc: Dict[str, np.ndarray])-> np.ndarray:
    labor_sims = sim_lc['lab'][:, :, :, :myPars.J]
    emp_sims = np.where(labor_sims > 0, 1, 0)
    weighted_emp_sims = model.gen_weighted_sim(myPars, emp_sims)
    mean_emp_by_age = np.sum(weighted_emp_sims, axis = tuple(range(weighted_emp_sims.ndim-1)))
    return mean_emp_by_age     

def plot_emp_aggs_and_moms(myPars : Pars, sim_lc: Dict[str, np.ndarray], data_moms_path: str = None, outpath: str = None)-> None:
    if outpath == None:
        outpath = myPars.path + 'output/'
    if data_moms_path == None:
        data_moms_path = myPars.path + '/input/emp_rate_moments.csv'
    avg_emp = weighted_emp_rate_by_age(myPars, sim_lc)
    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = avg_emp
    sim_key_label = "Simulated"
    data_moments_label = 'From the data'
    log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number
    data_moments_col_ind = 1
    data_moments = tb.read_specific_column_from_csv(data_moms_path, data_moments_col_ind) # 1 means read the second column
    log_moments = np.log(np.where(data_moments > 0, data_moments, 1e-3))
    for modifier in ['','log']:
        if modifier == 'log':
            sim_values = log_values
            mom_values = log_moments
            sim_y_label = "Average Log Employment Rate (Weighted)"
        else:
            sim_values = values
            mom_values = data_moments
            sim_y_label = "Average Employment Rate (Weighted)"
        if myPars.print_screen >= 2:
            print(sim_y_label)
        
        fig, ax = plt.subplots()
        ax.plot(age_grid, sim_values, label = sim_key_label)
        ax.plot(age_grid, mom_values, label = data_moments_label)
        # specify axis and labels
        ax.set_xlabel('Age')
        ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2])
        ax.set_ylabel(sim_y_label)
        ax.legend()

        short_name = 'emp_rate'

        #save the figure
        fullpath = outpath + f'fig_fit_{short_name}_{modifier}.pdf'
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        #save the data
        fullpath =  outpath + f'fig_fit_{short_name}_{modifier}.csv'
        with open(fullpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['age'] + list(age_grid))
            writer.writerow(['model'] + list(sim_values))
            writer.writerow(['data'] + list(data_moments))


def plot_H_trans_H_type_alg(myPars1: Pars, myPars2: Pars, path: str = None, 
                            low_type_out_file_name: str = None, high_type_out_file_name: str = None, 
                            plot_and_csv_name1: str = None, plot_and_csv_name2: str = None
                            ) -> Tuple[plt.Figure, plt.Axes, plt.Figure, plt.Axes]:
    if path is None:
        print("***path is none***")
        path = myPars1.path + 'output/'
    if plot_and_csv_name1 is None:
        plot_and_csv_name1 = "test_H_trans_by_H_type_alg1"
    if plot_and_csv_name2 is None:
        plot_and_csv_name2 = "test_H_trans_by_H_type_alg2"
    if low_type_out_file_name is None:
        low_type_out_file_name = "test_fig_low_type_H_trans"
    if high_type_out_file_name is None:
        high_type_out_file_name = "test_fig_high_type_H_trans"
    
    # Generate the first set of plots
    fig1, ax1 = plot_H_trans_H_type(myPars1, path, plot_and_csv_name1)
    fig2, ax2 = plot_H_trans_H_type(myPars2, path, plot_and_csv_name2)
    
    # Get lines from both plots
    lines1 = ax1.get_lines()
    lines2 = ax2.get_lines()
    
    # Create new figures and axes for the low and high type plots
    low_fig, low_ax = plt.subplots()
    high_fig, high_ax = plt.subplots()
    color_list = ['g', 'r', 'g','r']

    # Plot the first two lines on the low figure
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if i % 2 == 0:
            new_label1 = f"Bad to Good (k2)"
            new_label2 = f"Bad to Good (50p)"
        else:
            new_label1 = f"Good to Bad (k2)"
            new_label2 = f"Good to Bad (50p)"
        if i < 2:
            # Solid lines from fig1, dashed lines from fig2
            low_ax.plot(line1.get_xdata(), line1.get_ydata(), color= color_list[i], linestyle='-', label=new_label1)
            low_ax.plot(line2.get_xdata(), line2.get_ydata(), color= color_list[i], linestyle='--', label=new_label2)
        else:
            # Remaining lines go to the high figure
            high_ax.plot(line1.get_xdata(), line1.get_ydata(), color= color_list[i], linestyle='-', label=new_label1)
            high_ax.plot(line2.get_xdata(), line2.get_ydata(), color= color_list[i], linestyle='--', label=new_label2)

    # Set labels, limits, and legends for both figures
    for ax in (low_ax, high_ax):
        ax.set_xlabel('Age')
        ax.set_xlim([myPars1.age_grid[0] - 2, myPars1.age_grid[-1] + 2])
        ax.set_ylabel('Probability (%)')
        ax.set_ylim([0, 1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncols=2)

    low_ax.title.set_text('Low Type Mental Health Transitions by Typing Alg.')
    high_ax.title.set_text('High Type Mental Health Transitions by Typing Alg.')
    # Save and display the low-type plot
    low_path = path + f'fig_{low_type_out_file_name}.pdf'
    low_fig.savefig(low_path, bbox_inches='tight')
    plt.show()

    # Save and display the high-type plot
    high_path = path + f'fig_{high_type_out_file_name}.pdf'
    high_fig.savefig(high_path, bbox_inches='tight')
    plt.show()

    return low_fig, low_ax, high_fig, high_ax

def plot_H_trans_H_type(myPars: Pars, path: str = None, plot_and_csv_name: str = None,
                        colors: list[str] = None)-> Tuple[plt.Figure, plt.Axes]:
    if path == None:
        path = myPars.path + 'output/'
    #if the path doesn't exist, create it
    if not os.path.exists(path):
        os.makedirs(path)

    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = myPars.H_trans
    y_label = "Probability (%)"
    fig, ax = plt.subplots()

    if plot_and_csv_name == None:
        plot_and_csv_name = "H_trans_by_H_type"
    csv_path = path + f'{plot_and_csv_name}.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['age'] + list(age_grid))

    
    for H_type_perm_ind in range(myPars.H_type_perm_grid_size):
        for curr_H_state in range(myPars.H_grid_size):
            for fut_H_state in range(myPars.H_grid_size):
                #we want transitions that are H changes
                if curr_H_state != fut_H_state:
                    trans = values[H_type_perm_ind,:,curr_H_state,fut_H_state] 
                    lab = f"{curr_H_state} to {fut_H_state}, u_{{H}} = {myPars.H_type_perm_grid[H_type_perm_ind]}"
                    ax.plot(age_grid, trans, label = lab)
                    #save the data
                    with open(csv_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Transitions"] + list(trans))

    ax.set_xlabel('Age')
    ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2])
    ax.set_ylabel(y_label)
    ax.set_ylim([0, 1])
    ax.legend()

    #save the figure
    fullpath = path + f'fig_{plot_and_csv_name}.pdf'
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close(fig)
    return fig, ax

def plot_H_trans_uncond(myPars: Pars, path: str = None, plot_and_csv_name: str = None)-> None:
    if path == None:
        path = myPars.path + 'output/'
    if not os.path.exists(path):
        os.makedirs(path)

    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = myPars.H_trans 
    y_label = "Health Transition Probability"
    fig, ax = plt.subplots()

    if plot_and_csv_name == None:
        plot_and_csv_name = "H_trans_uncond"
    csv_path = path + f'{plot_and_csv_name}.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['age'] + list(age_grid))

    for curr_H_state in range(myPars.H_grid_size):
        for fut_H_state in range(myPars.H_grid_size):
            #we want transitions that are H changes
            if curr_H_state != fut_H_state:
                
                trans = values[0,:,curr_H_state,fut_H_state] 
                if curr_H_state == 0:
                    lab = f"MH Recovery: Bad to Good"
                    color = 'g'
                else:
                    lab = f"MH Decline: Good to Bad"
                    color = 'r'
                ax.plot(age_grid, trans, label = lab, color = color)
                #save the data
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Transitions"] + list(trans))

    ax.set_xlabel('Age')
    ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2])
    ax.set_ylabel(y_label)
    ax.set_ylim([0, 1])
    ax.legend()

    #save the figure
    fullpath = path + f'fig_{plot_and_csv_name}.pdf'
    fig.savefig(fullpath, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    start_time = time.perf_counter()