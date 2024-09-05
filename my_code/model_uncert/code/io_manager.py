"""
io_manager.py
This file contains code used to read in data and cluster it based on a history of mental health data.
It also contains code to print parameters and transition matrices to LaTeX tables.
And to print the results of the calibration to a LaTeX table.

Created on 2024-08-12 14:47:07 

by @author Ben Boyajian

"""
# imports
# general
import numpy as np
import csv 
import os
from typing import Dict, Tuple

# my code
import my_toolbox as tb
from pars_shocks import Pars, Shocks

def get_H_type_pop_shares(myPars: Pars, input_csv_path: str)-> Tuple[np.ndarray, np.ndarray]:
    """
    read in data for myPars.H_beg_pop_weights_by_H_type and myPars.H_type_perm_weights from input_csv_path
    return H_beg_pop_weights, type_pop_share
    """
    pop_share_path = input_csv_path
    H_beg_pop_weights = tb.read_specific_row_from_csv(pop_share_path, 0)[myPars.H_type_perm_grid_size:].reshape(myPars.H_type_perm_grid_size, myPars.H_grid_size)
    type_pop_share = tb.read_matrix_from_csv(pop_share_path, column_index = 0)[:myPars.H_type_perm_grid_size]
    return H_beg_pop_weights, type_pop_share

def read_and_shape_H_trans_full(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the full by age by type transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_by_MH_clust_age.csv"
    raw_mat = tb.read_matrix_from_csv(path)
    H_trans_mat_size_j = myPars.H_grid_size * myPars.H_grid_size   
    mat_sep_groups = raw_mat.reshape(myPars.J,  H_trans_mat_size_j, myPars.H_type_perm_grid_size) 
    final_reshape = mat_sep_groups.reshape(myPars.J, myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size).transpose(1, 0, 2, 3)
    return final_reshape

def read_and_shape_H_trans_uncond_age(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the transition matrix for the health state conditional on age but not type
    """
    if path is None:
        path = myPars.path + "input/MH_trans_uncond_age.csv"
    raw_mat = tb.read_matrix_from_csv(path)
    reshaped_mat = raw_mat.reshape(myPars.J, myPars.H_grid_size, myPars.H_grid_size) # Ensure mat is a 3D array
    repeated_mat = np.repeat(np.array(reshaped_mat)[np.newaxis, :, :, :], myPars.H_type_perm_grid_size, axis = 0)
    return repeated_mat

def read_and_shape_H_trans_uncond(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the unconditional on type and age transition matrix for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_uncond.csv"
    mat = tb.read_specific_row_from_csv(path, 0)
    mat = np.array(mat).reshape(myPars.H_grid_size, myPars.H_grid_size)  # Ensure mat is a 2D array
    repeated_matrix = np.tile(mat, (2, 1, 1))
    H_trans = np.repeat(np.array(repeated_matrix)[:, np.newaxis, :,:], myPars.J, axis=0).reshape(myPars.H_type_perm_grid_size, myPars.J,
                                                                                                 myPars.H_grid_size, myPars.H_grid_size)
    return H_trans

def read_and_shape_H_trans_H_type(myPars: Pars, path: str = None) -> np.ndarray:
    """
    Read in the by type transition matrix (UNCONDITIONAL ON AGE) for the health state and reshape it to the correct dimensions
    """
    if path is None:
        path = myPars.path + "input/MH_trans_by_MH_clust.csv"
    raw_mat = tb.read_specific_row_from_csv(path, 0)
    raw_mat = np.array(raw_mat).reshape(myPars.H_type_perm_grid_size, myPars.H_grid_size, myPars.H_grid_size)  # Ensure mat is a 3D array
    H_trans = np.repeat(np.array(raw_mat)[:, np.newaxis, :,:], myPars.J, axis=0).reshape(myPars.H_type_perm_grid_size,myPars.J,
                                                                                         myPars.H_grid_size, myPars.H_grid_size)
    return H_trans

def print_endog_params_to_tex(myPars: Pars, targ_moments: Dict[str, float], model_moments: Dict[str, float], path: str = None) -> None:
    '''This generates a LaTeX table of the parameters and compiles it to a PDF.'''
    alpha_targ_val = targ_moments['alpha']*100
    alpha_mod_val = model_moments['alpha']*100
    w1_targ_val = targ_moments['w1']*100
    w1_mod_val = model_moments['w1']*100
    w2_targ_val = targ_moments['w2']*100
    w2_mod_val = model_moments['w2']*100
    wH_targ_val = targ_moments['wH']*100
    wH_mod_val = model_moments['wH']*100

    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\begin{document}\n",
        "\\small\n",
        "\\begin{tabular}{l l l l l l} \n",
        "\\hline \n",
        "Parameter & Description & Par. Value & Target Moment & Target Value & Model Value \\\\ \n", 
        "\\hline \n",   
        f"$\\alpha$ & $c$ utility weight & {round(myPars.alpha, 4)} & Mean hours worked & {round(alpha_targ_val,2)} & {round(alpha_mod_val, 2)} \\\\ \n", 
        f"$w_{{1}}$ & Linear wage coeff. & {round(myPars.wage_coeff_grid[1,1], 4)} & Wage growth & {round(w1_targ_val,2)}\\% & {round(w1_mod_val, 2)}\\% \\\\ \n", 
        f"$w_{{2}}$ & Quad. wage coeff. & {round(myPars.wage_coeff_grid[1,2], 4)} & Wage decay & {round(w2_targ_val,2)}\\% & {round(w2_mod_val,2)}\\% \\\\ \n", 
        f"$w_{{H}}$ & Health wage coeff. & {round(myPars.wH_coeff, 4)} & Healthy wage premium & {round(wH_targ_val,2)}\\% & {round(wH_mod_val,2)}\\% \\\\ \n", 
        "\\hline \n",
        "\\end{tabular}\n",
        "\\end{document}\n"
    ]
    
    if path is None:
        path = myPars.path + 'output/'
    file_name = 'parameters_endog.tex'
    tb.list_to_tex(path, file_name, tab)
    tb.tex_to_pdf(path, file_name)
    
def print_w0_calib_to_tex(myPars: Pars, targ_moments: Dict[str, float], model_moments: Dict[str, float], path: str = None) -> None:
    '''This generates a LaTeX table of the parameters and compiles it to a PDF.'''
    w0_mean_targ_val = np.round(targ_moments['w0_mean'], 3)
    w0_mean_mod_val = np.round(model_moments['w0_mean'], 3)
    w0_sd_targ_val = np.round(targ_moments['w0_sd'], 3)
    w0_sd_mod_val = np.round(model_moments['w0_sd'], 3)

    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\begin{document}\n",
        "\\small\n",
        "\\begin{tabular}{l l l l} \n",
        "\\hline \n",
        "Constant wage coeff. & Ability Level & Value & Weight \\\\ \n",
        "\\hline \n",
        f"$w_{{0\\gamma_{{1}}}}$ & Low & {round(np.exp(myPars.wage_coeff_grid[0, 0]))} & {round(myPars.lab_fe_weights[0],2)} \\\\ \n",
        f"$w_{{0\\gamma_{{2}}}}$ & Medium & {round(np.exp(myPars.wage_coeff_grid[1, 0]))} & {round(myPars.lab_fe_weights[1],2)} \\\\ \n",
        f"$w_{{0\\gamma_{{3}}}}$ & Medium High & {round(np.exp(myPars.wage_coeff_grid[2, 0]))} & {round(myPars.lab_fe_weights[2],2)} \\\\ \n",
        f"$w_{{0\\gamma_{{4}}}}$ & High & {round(np.exp(myPars.wage_coeff_grid[3, 0]))} & {round(myPars.lab_fe_weights[3],2)} \\\\ \n",
        "\\hline \n",
        "Target Moment & Target Value & Model Value & \\\\ \n",
        "\\hline \n",
        f"Mean wage, $j=0$ & {w0_mean_targ_val} & {w0_mean_mod_val} & \\\\ \n",
        f"SD wage, $j=0$ & {w0_sd_targ_val} & {w0_sd_mod_val} & \\\\ \n",
        "\\hline \n",
        "\\end{tabular}\n",
        "\\end{document}\n"
    ]

    if path is None:
        path = myPars.path + 'output/'
    tex_file_name =  'parameters_w0_calib.tex' 
    tb.list_to_tex(path, tex_file_name, tab)
    tb.tex_to_pdf(path, tex_file_name)

def table_H_trans_by_type_alg(myPars: Pars, H_trans_alg_0: np.ndarray, H_trans_alg_1: np.ndarray, 
                              out_path: str  = None, tex_file_name: str = None, 
                              )-> None:
    """prints transition matrices by typing method to a LaTeX table and compiles it to a PDF"""
    if out_path is None:
        out_path = myPars.path + 'output/'
    if tex_file_name is None:
        tex_file_name = "H_trans_by_type_alg_test.tex"
    
    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\usepackage{booktabs}",  # Ensure booktabs package is loaded
        "\\begin{document}\n",
        # "\\small\n",
        # "\\caption{Transition Matrices by Typing Method} \n",
        "\\begin{tabular}{l l l l l l l} \n",
        "\\hline \n",
        "\\hline \n",
        "\\\\[0.5mm] \n",
        "Typing Method & Low Type & Bad & Good & High Type & Bad & Good \\\\ \n",
        "\\cmidrule(lr){1-1} \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \n",
        f"50pth Cutoff  & Bad & {round(H_trans_alg_0[0,0,0], 3)} & {round(H_trans_alg_0[0,0,1], 3)} & Bad & {round(H_trans_alg_0[1,0,0], 3)} & {round(H_trans_alg_0[1,0,1], 3)} \\\\ \n",
        f"50pth Cutoff  & Good & {round(H_trans_alg_0[0,1,0], 3)} & {round(H_trans_alg_0[0,1,1], 3)} & Good & {round(H_trans_alg_0[1,1,0], 3)} & {round(H_trans_alg_0[1,1,1], 3)} \\\\ \n",
        "\\\\[1mm] \n",
        "Typing Method & Low Type & Bad & Good & High Type & Bad & Good \\\\ \n",
        "\\cmidrule(lr){1-1} \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \n",
        f"$k$-means$\\left(k=2\\right)$ & Bad & {round(H_trans_alg_1[0,0,0], 3)} & {round(H_trans_alg_1[0,0,1], 3)} & Bad & {round(H_trans_alg_1[1,0,0], 3)} & {round(H_trans_alg_1[1,0,1], 3)} \\\\ \n",
        f"$k$-means$\\left(k=2\\right)$  & Good & {round(H_trans_alg_1[0,1,0], 3)} & {round(H_trans_alg_1[0,1,1], 3)} & Good & {round(H_trans_alg_1[1,1,0], 3)} & {round(H_trans_alg_1[1,1,1], 3)} \\\\ \n",
        "\\\\[0.5mm] \n",
        "\\hline \n",
        "\\hline \n",
        "Some text for a footnote? \n",
        "\\end{tabular}\n",
        "\\end{document}\n"
    ]

    tb.list_to_tex(out_path, tex_file_name, tab)
    tb.tex_to_pdf(out_path, tex_file_name)

def table_r2_by_type_alg(myPars: Pars, r2_arr: np.ndarray, 
                              out_path: str  = None, tex_file_name: str = None, 
                              )-> None:
    """This function prints the R^2 values for the different typing methods to a LaTeX table and compiles it to a PDF."""
    if out_path is None:
        out_path = myPars.path + 'output/'
    if tex_file_name is None:
        tex_file_name = "table_r2_by_type_alg.tex"
    
    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\usepackage{booktabs}",  # Ensure booktabs package is loaded
        "\\begin{document}\n",
        # "\\small\n",
        # "\\caption{Transition Matrices by Typing Method} \n",
        "\\begin{tabular}{l l l l l l l} \n",
        "\\hline \n",
        "\\midrule \n",
        "Lagged MH     & x &   &   & x & x & - \\\\ \n",
        "MH Type 50pth &   & x &   & x &   & - \\\\ \n",
        "MH Type $k$-means$\\left(k=2\\right)$    &   &   & x &   & x & - \\\\ \n",
        "\\midrule \n",
        f"""$R^{{2}}$               & {round(r2_arr[0][0], 3)}  & {round(r2_arr[0][1], 3)} & {round(r2_arr[0][2], 3)} 
                                    & {round(r2_arr[0][3], 3)}  & {round(r2_arr[0][4], 3)} & - \\\\ \n""",
        f"""$R^{{2}}$ with controls & {round(r2_arr[1][0], 3)}  & {round(r2_arr[1][1], 3)} & {round(r2_arr[1][2], 3)} 
                                    & {round(r2_arr[1][3], 3)}  & {round(r2_arr[1][4], 3)} & {round(r2_arr[1][5], 3)} \\\\ \n""",
        "\\midrule \n",
        "\\hline \n",
        "Some text for a footnote? \n",
        "\\end{tabular}\n",
        "\\end{document}\n"
    ]
    tb.list_to_tex(out_path, tex_file_name, tab)
    tb.tex_to_pdf(out_path, tex_file_name)

def print_wage_coeffs_to_tex(myPars: Pars, path: str = None)-> None:
    '''this generates a latex table of the parameters'''
    tab = ["\\small\\begin{tabular}{l l l l l l} \n"]        
    tab.append("\\hline \n")
    tab.append(" Parameter & $\\gamma_1$ &  $\\gamma_2$ & $\\gamma_3$ & $\\gamma_4$ & Description & Source \\\\ \n") 
    tab.append("\\hline \n")   
    tab.append(f"""$w_{{0\\gamma}}$ & {round(myPars.wage_coeff_grid[0][0], 3)} & {round(myPars.wage_coeff_grid[1][0], 3)} 
               & {round(myPars.wage_coeff_grid[2][0], 3)} 
               & Constant & Benchmark \\\\ \n""")
    tab.append(f"""$w_{{1\\gamma}}$ & {round(myPars.wage_coeff_grid[0][1], 3)} & {round(myPars.wage_coeff_grid[1][1], 3)} 
               & {round(myPars.wage_coeff_grid[2][1], 3)} 
               & $j$ Coeff. & Wage Growth \\\\ \n""")
    tab.append(f"""$w_{{2\\gamma}}$ & {round(myPars.wage_coeff_grid[0][2], 3)} & {round(myPars.wage_coeff_grid[1][2], 3)} 
               & {round(myPars.wage_coeff_grid[2][2], 3)}  
               & $j^{{2}}$ Coeff. & Wage Decline \\\\ \n""")
    tab.append("\\hline \n")
    tab.append(f"\\end{{tabular}}")
    
    if path is None:
        path = myPars.path + 'output/'
    tex_file_name = 'wage_coeffs.tex'
    tb.list_to_tex(path, tex_file_name, tab)
    tb.tex_to_pdf(path, tex_file_name)

def print_H_trans_to_tex_uncond(myPars: Pars, out_path: str = None, tex_file_name: str = None)-> None:
    """This function prints the unconditional transition matrix to a LaTeX table and compiles it to a PDF."""
    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\usepackage{amsmath}\n",  # Added this line for better array formatting
        "\\begin{document}\n",
        "\\[ \\left[\\begin{array}{cc} \n",
        f"{round(myPars.H_trans[0, 0, 0, 0], 2)}, & {round(myPars.H_trans[0, 0, 0, 1], 2)} \\\\ \n",
        f"{round(myPars.H_trans[0, 0, 1, 0], 2)}, & {round(myPars.H_trans[0, 0, 1, 1], 2)} \n"
        "\\end{array}\\right] \\] \n",
        "\\end{document}"
    ]

    if out_path is None:
        out_path = myPars.path + 'output/'
    if tex_file_name is None:
        tex_file_name = 'H_trans_uncond.tex'
    tb.list_to_tex(out_path, tex_file_name, tab)
    tb.tex_to_pdf(out_path, tex_file_name)

def print_H_trans_to_tex(myPars: Pars, trans_matrix: np.ndarray, out_path: str = None, new_file_name: str = None, 
                         tex_lhs_of_equals: str = None)-> None:
    if trans_matrix.shape != (2, 2):
        raise ValueError("Transition matrix must be 2x2.")
    
    if tex_lhs_of_equals is None:
        tex_lhs_of_equals =  f"\\Pi_{{H}}"

    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\usepackage{amsmath}\n",  # Added this line for better array formatting
        "\\begin{document}\n",
        "\\[\n",  # Added this line to start the LaTeX math environment 
        tex_lhs_of_equals,
        "= \n",
        "\\left[\\begin{array}{cc} \n",
        f"{round(trans_matrix[0, 0], 2)} & {round(trans_matrix[0, 1], 2)} \\\\ \n",
        f"{round(trans_matrix[1, 0], 2)} & {round(trans_matrix[1, 1], 2)} \n",
        "\\end{array}\\right] \\] \n",
        "\\end{document}"
    ]

    if out_path is None:
        out_path = myPars.path + 'output/'  # Assuming `myPars.path` is available globally
    if new_file_name is None:
        new_file_name = 'H_trans_test.tex'
    else:
        new_file_name = new_file_name + '.tex'
    tb.list_to_tex(out_path, new_file_name, tab)
    tb.tex_to_pdf(out_path, new_file_name)


def print_exog_params_to_tex(myPars: Pars, path: str = None)-> None:
    '''this generates a latex table of the parameters'''
    tab = ["\\documentclass[border=3mm,preview]{standalone}",
            "\\begin{document}\n",
            "\\small\n",
            "\\begin{tabular}{l l l l} \n"
            "\\hline \n",
            "Parameter & Description & Value & Source \\\\ \n",
            "\\hline \n",
            f"$R$ & Gross interest rate  & {np.round(1 + myPars.r, 4)} & Benchmark \\\\ \n",
            f"$\\beta$ & Patience & {np.round(myPars.beta, 4)} & $1/R$ \\\\ \n",
            f"$\\sigma$ & CRRA & {np.round(myPars.sigma_util, 4)} & Benchmark \\\\ \n",
            f"$\\phi_n$ & Labor time-cost & {np.round(myPars.phi_n, 4)} & Benchmark \\\\ \n",
            f"$\\phi_H$ & Health time-cost & {np.round(myPars.phi_H, 4)} & Benchmark \\\\ \n",
            f"$\\omega_{{H=0}}$ & Low type pop. weight & {np.round(myPars.H_type_perm_weights[0], 4)} & UKHLS \\\\ \n",
            f"$\\omega_{{H=1}}$ & High type pop. weight & {np.round(myPars.H_type_perm_weights[1], 4)} & $1-\\omega_{{H=0}}$ \\\\ \n",
            "\\hline \n",
            "\\end{tabular}\n",
            "\\end{document}\n"
            ]
    if path is None:
        path = myPars.path + 'output/'
    tex_file_name = 'parameters_exog.tex'
    tb.list_to_tex(path, tex_file_name, tab)
    tb.tex_to_pdf(path, tex_file_name)


def print_params_to_csv(myPars: Pars, path: str = None, file_name: str = "parameters.csv")-> None:
    """
    prints the parametes from myPars to a csv file
    takes in the path and file name with a .csv extension
    """
    if path is None:
        path = myPars.path + 'output/calibration/'
    else:
        path = path + 'calibration/'
    if not os.path.exists(path):
        os.makedirs(path)
    my_path = path + file_name
    with open(my_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        for param, value in pars_to_dict(myPars).items():
            writer.writerow([param, value])

def pars_to_dict(pars_instance: Pars) -> Dict:
    return {
        'sigma_gamma_2': pars_instance.sigma_gamma_2,
        'lab_fe_grid': pars_instance.lab_fe_grid,
        'lab_fe_grid_size': pars_instance.lab_fe_grid_size,
        'beta': pars_instance.beta,
        'alpha': pars_instance.alpha,
        'sigma_util': pars_instance.sigma_util,
        'phi_n': pars_instance.phi_n,
        'phi_H': pars_instance.phi_H,
        'r': pars_instance.r,
        'a_min': pars_instance.a_min,
        'a_max': pars_instance.a_max,
        'a_grid_growth': pars_instance.a_grid_growth,
        'a_grid': pars_instance.a_grid,
        'a_grid_size': pars_instance.a_grid_size,
        'H_grid': pars_instance.H_grid,
        'H_grid_size': pars_instance.H_grid_size,
        'state_space_shape': pars_instance.state_space_shape,
        'state_space_shape_no_j': pars_instance.state_space_shape_no_j,
        'state_space_no_j_size': pars_instance.state_space_no_j_size,
        'state_space_shape_sims': pars_instance.state_space_shape_sims,
        'lab_min': pars_instance.lab_min,
        'lab_max': pars_instance.lab_max,
        'c_min': pars_instance.c_min,
        'leis_min': pars_instance.leis_min,
        'leis_max': pars_instance.leis_max,
        'sim_draws': pars_instance.sim_draws,
        'J': pars_instance.J,
        'print_screen': pars_instance.print_screen,
        'interp_c_prime_grid': pars_instance.interp_c_prime_grid,
        'interp_eval_points': pars_instance.interp_eval_points,
        'sim_interp_grid_spec': pars_instance.sim_interp_grid_spec,
        'start_age': pars_instance.start_age,
        'end_age': pars_instance.end_age,
        'age_grid': pars_instance.age_grid,
        'path': pars_instance.path,
        'wage_coeff_grid': pars_instance.wage_coeff_grid,
        'wH_coeff': pars_instance.wH_coeff,
        'wage_min': pars_instance.wage_min,
        'max_iters': pars_instance.max_iters,
        'max_calib_iters': pars_instance.max_calib_iters,
    }



# run if main function
if __name__ == "__main__":

    path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    myPars = Pars(path, J=51)

