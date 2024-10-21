"""
model_uncert - tables

Author: Ben Boyajian
Date: 2024-09-05 11:23:18 
"""
#Import packages
#General
from typing import Dict
import numpy as np
#My code
from pars_shocks import Pars
import my_toolbox as tb


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
    w0_mu_targ_val = targ_moments['w0_mu']
    w0_mu_mod_val = model_moments['w0_mu']
    w0_sigma_targ_val = targ_moments['w0_sigma']
    w0_sigma_mod_val = model_moments['w0_sigma']

    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\begin{document}\n",
        "\\small\n",
        "\\begin{tabular}{l l l l l l} \n",
        "\\hline \n",
        "Parameter & Description & Par. Value & Target Moment & Target Value & Model Value \\\\ \n", 
        "\\hline \n",   
        f"$\\alpha$ & $c$ utility weight & {round(myPars.alpha, 4)} & Mean hours worked & {round(alpha_targ_val,2)} & {round(alpha_mod_val, 2)} \\\\ \n", 
        f"$\\mu_{{w_{{0}}}}$ & FE wage mean & {round(myPars.lab_fe_tauch_mu, 4)} & Mean wage, $j=0$ & {round(w0_mu_targ_val,4)} & {round(w0_mu_mod_val,4)} \\\\ \n",
        f"$\\sigma_{{w_{{0}}}}$ & FE wage SD & {round(myPars.lab_fe_tauch_sigma, 4)} & SD wage, $j=0$ & {round(w0_sigma_targ_val,4)} & {round(w0_sigma_mod_val,4)} \\\\ \n",
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
    w0_mean_targ_val = np.round(targ_moments['w0_mu'], 3)
    w0_mean_mod_val = np.round(model_moments['w0_mu'], 3)
    w0_sd_targ_val = np.round(targ_moments['w0_sigma'], 3)
    w0_sd_mod_val = np.round(model_moments['w0_sigma'], 3)

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

def print_cum_earn_moms(myPars: Pars, model_moms: Dict[str, float], data_moms: Dict[str, float], outpath: str = None, 
                            tex_file_name: str = None) -> None:
    '''This generates a LaTeX table of the cumulative earnings moments and compiles it to a PDF.'''
    if outpath is None:
        outpath = myPars.path + 'output/'
    if tex_file_name is None:
        tex_file_name = 'cum_earns_moms.tex'
    
    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\usepackage{booktabs}",  # Ensure booktabs package is loaded
        "\\begin{document}\n",
        "\\textit{Cumulative Earnings Moments} \\\\ \n",
        "\\begin{tabular}{l l l l l l l} \n",
        "\\toprule \n",
        "Source & Group & Mean (logs) & SD (logs) & 90p/10p & 90p/50p & 50p/10p \\\\ \n",
        "\\midrule \n",
        # "Model & H = 1 &  &  &  &  &  \\\\ \n",
        # "Model & H = 0 &  &  &  &  &  \\\\ \n",
        f"""Model & Overall & {round(model_moms["mean"], 2)}  & {round(model_moms["sd"],2)} 
                            & {round(model_moms["90_10"], 2)}  & {round(model_moms["90_50"],2)} 
                            & {round(model_moms["50_10"], 2)}  \\\\ \n""",
        "\\midrule \n",
        # "Data & H = 1 &  &  &  &  &  \\\\ \n",
        # "Data & H = 0 &  &  &  &  &  \\\\ \n",
        # "Data & Overall &  &  &  &  &  \\\\ \n",
        f"""Data & Overall & {round(data_moms["mean"], 2)}  & {round(data_moms["sd"],2)}
                            & {round(data_moms["90_10"], 2)}  & {round(data_moms["90_50"],2)}
                            & {round(data_moms["50_10"], 2)}  \\\\ \n""",
        "\\bottomrule \n",
        "\\end{tabular}\n",
        "\\end{document}\n"
    ]

    tb.list_to_tex(outpath, tex_file_name, tab)
    tb.tex_to_pdf(outpath, tex_file_name)

def table_H_trans_by_type_alg(myPars: Pars, H_trans_alg_0: np.ndarray, H_trans_alg_1: np.ndarray, 
                              outpath: str  = None, tex_file_name: str = None, 
                              )-> None:
    """prints transition matrices by typing method to a LaTeX table and compiles it to a PDF"""
    if outpath is None:
        outpath = myPars.path + 'output/'
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
        "Typing Method & Low Type$\\left(50.0\\%\\right)$  & Bad & Good & High Type$\\left(50.0\\%\\right)$ & Bad & Good \\\\ \n",
        "\\cmidrule(lr){1-1} \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \n",
        f"50pth Cutoff  & Bad & {round(H_trans_alg_0[0,0,0], 3)} & {round(H_trans_alg_0[0,0,1], 3)} & Bad & {round(H_trans_alg_0[1,0,0], 3)} & {round(H_trans_alg_0[1,0,1], 3)} \\\\ \n",
        f"50pth Cutoff  & Good & {round(H_trans_alg_0[0,1,0], 3)} & {round(H_trans_alg_0[0,1,1], 3)} & Good & {round(H_trans_alg_0[1,1,0], 3)} & {round(H_trans_alg_0[1,1,1], 3)} \\\\ \n",
        "\\\\[1mm] \n",
        f"Typing Method & Low Type$\\left({round(myPars.H_type_perm_weights[0]*100,1)}\\%\\right)$ & Bad & Good & High Type$\\left({round(myPars.H_type_perm_weights[1]*100,1)}\\%\\right)$ & Bad & Good \\\\ \n",
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


    tb.list_to_tex(outpath, tex_file_name, tab)
    tb.tex_to_pdf(outpath, tex_file_name)

def table_r2_by_type_alg(myPars: Pars, r2_arr: np.ndarray, 
                              outpath: str  = None, tex_file_name: str = None, 
                              )-> None:
    """This function prints the R^2 values for the different typing methods to a LaTeX table and compiles it to a PDF."""
    if outpath is None:
        outpath = myPars.path + 'output/'
    if tex_file_name is None:
        tex_file_name = "table_r2_by_type_alg.tex"
    
    tab = [
        "\\documentclass[border=3mm,preview]{standalone}",
        "\\usepackage{booktabs}",  # Ensure booktabs package is loaded
        "\\begin{document}\n",
        
        # Add a title or descriptive text at the top to declare the outcome variable
        "\\textit{Outcome Variable: Mental Health Index (SF-12)} \\\\ \n",
        # "\\textit{This table presents regression results for the specified outcome variable.} \\\\ \n",
        
        # Start of the table
        "\\begin{tabular}{l l l l l l l} \n",
        "\\toprule \n",  # Use \toprule for the top line instead of \hline and \midrule
        "Lagged MH     & x &   &   & x & x & - \\\\ \n",
        "MH Type 50pth &   & x &   & x &   & - \\\\ \n",
        "MH Type k-means ($k=2$)    &   &   & x &   & x & - \\\\ \n",
        "\\midrule \n",
        f"""$R^{{2}}$               & {round(r2_arr[0][0], 3)}  & {round(r2_arr[0][1], 3)} & {round(r2_arr[0][2], 3)} 
                                    & {round(r2_arr[0][3], 3)}  & {round(r2_arr[0][4], 3)} & - \\\\ \n""",
        f"""$R^{{2}}$ with controls & {round(r2_arr[1][0], 3)}  & {round(r2_arr[1][1], 3)} & {round(r2_arr[1][2], 3)} 
                                    & {round(r2_arr[1][3], 3)}  & {round(r2_arr[1][4], 3)} & {round(r2_arr[1][5], 3)} \\\\ \n""",
        "\\bottomrule \n",  # Use \bottomrule for the bottom line instead of \hline
        "Some text for a footnote. \n",
        "\\end{tabular}\n",
        "\\end{document}\n"
    ]

    tb.list_to_tex(outpath, tex_file_name, tab)
    tb.tex_to_pdf(outpath, tex_file_name)

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

def print_H_trans_to_tex_uncond(myPars: Pars, outpath: str = None, tex_file_name: str = None)-> None:
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

    if outpath is None:
        outpath = myPars.path + 'output/'
    if tex_file_name is None:
        tex_file_name = 'H_trans_uncond.tex'
    tb.list_to_tex(outpath, tex_file_name, tab)
    tb.tex_to_pdf(outpath, tex_file_name)

def print_H_trans_to_tex(myPars: Pars, trans_matrix: np.ndarray, outpath: str = None, new_file_name: str = None, 
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

    if outpath is None:
        outpath = myPars.path + 'output/'  # Assuming `myPars.path` is available globally
    if new_file_name is None:
        new_file_name = 'H_trans_test.tex'
    else:
        new_file_name = new_file_name + '.tex'
    tb.list_to_tex(outpath, new_file_name, tab)
    tb.tex_to_pdf(outpath, new_file_name)

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

