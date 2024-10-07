"""
Created on 2024-05-18 00:27:18
@author: Ben Boyaian
"""

# my_toolbox
# heavily bsed on toolbox.py from Adam Blandin
# editing by Ben Boyajian began 2024-05-18 00:27:18

import numpy as np
from numba import njit, guvectorize, prange
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.axes import Axes 
import time
from typing import List, Dict, Tuple, Callable, Optional
import os
import subprocess
from scipy.optimize import minimize, differential_evolution

@njit
def range_tuple_numba(length: int) -> Tuple[int]:
    """
    returns a tuple of integers from 0 to length
    """
    if length == 0:
        return ()
    elif length == 1:
        return (0,)
    elif length == 2:
        return (0, 1)
    elif length == 3:
        return (0, 1, 2)
    elif length == 4:
        return (0, 1, 2, 3)
    elif length == 5:
        return (0, 1, 2, 3, 4)
    elif length == 6:
        return (0, 1, 2, 3, 4, 5)
    elif length == 7:
        return (0, 1, 2, 3, 4, 5, 6)
    elif length == 8:
        return (0, 1, 2, 3, 4, 5, 6, 7)
    elif length == 9:
        return (0, 1, 2, 3, 4, 5, 6, 7, 8)
    elif length == 10:
        return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def collapse_to_last_dim_wperc(values_array: np.ndarray, weights: np.ndarray, percentile: float) -> np.ndarray:
    """
    Collapse the values_array to the last dimension by taking the weighted percentile 
    across all other dimensions.
    
    Parameters:
    -----------
    values_array: np.ndarray
        A multi-dimensional array where we want to collapse all dimensions except the last one.
    weights: np.ndarray
        A weights array with one less dimension than values_array, applied uniformly across the last dimension.
    percentile: float
        The desired percentile (0 to 100) to calculate, e.g., 50 for the median.
        
    Returns:
    --------
    np.ndarray
        A 1D array containing the weighted percentiles collapsed across all dimensions except the last one.
    """
    
    # Ensure the percentile is between 0 and 100
    assert 0 <= percentile <= 100, "Percentile must be between 0 and 100"
    values_array_shape_min1 = values_array.shape[:len(values_array.shape)-1]
    if values_array_shape_min1 != weights.shape:
        raise ValueError("values_array and weights must have the same shape except for the last dimension")
    
    # Shape of the output should be equal to the size of the last dimension
    output_shape = (values_array.shape[-1],)
    result = np.zeros(output_shape)
    
    # Iterate over each index in the last dimension
    for idx in range(values_array.shape[-1]):
        # Extract the slice of values and weights for the current index in the last dimension
        sub_array = values_array[..., idx]
        sub_weights = weights
        
        # Flatten both the sub_array and sub_weights
        flattened_values = sub_array.flatten()
        flattened_weights = sub_weights.flatten()
        
        # Sort the values and weights by the values
        sorted_indices = np.argsort(flattened_values)
        sorted_values = flattened_values[sorted_indices]
        sorted_weights = flattened_weights[sorted_indices]

        # Compute cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        total_weight = cumulative_weights[-1]
        
        # Determine the percentile position
        percentile_position = percentile / 100 * total_weight
        
        # Find the value at the specified percentile
        idx_percentile = np.searchsorted(cumulative_weights, percentile_position)
        
        # Store the value corresponding to the calculated percentile
        result[idx] = sorted_values[idx_percentile]
    
    return result

@njit
def mean_nonzero_numba(arr: np.ndarray) -> float:
    """
    takes a numpy array and returns the mean of the non-zero elements
    """
    total = 0.0
    count = 0
    # Flatten the array for easier iteration
    flat_arr = arr.flatten()
    
    for i in range(flat_arr.size):
        if flat_arr[i] != 0:
            total += flat_arr[i]
            count += 1
    
    # If no non-zero elements, return 0 to avoid division by zero
    if count == 0:
        return 0.0
    return total / count

def plot_lc_mom_by_age(lc_mom_by_age: np.ndarray, age_grid: np.ndarray, mom_name: str, save_path: str = None, quietly: bool = False) -> Tuple[Figure, Axes]:
    J = len(age_grid) - 1
    values = lc_mom_by_age[:J]
    age_grid = age_grid[:J]
    # print("values.shape", lc_mom_by_age.shape)
    # print("age_grid.shape", age_grid.shape)
    y_label = mom_name
    key_label = mom_name
    x_label = "Age"

    fig, ax = plt.subplots()
    ax.plot(age_grid, values, label = key_label)
    ax.set_xlabel(x_label)
    ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2])
    ax.set_ylabel(y_label)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if not quietly:
        plt.show()
    plt.close()

    return fig, ax

def combine_plots(
    figures_axes: List[Tuple[Figure, Axes]], 
    save_path: Optional[str] = None,
    comb_fig_title: Optional[str] = None,
    x_lim: Optional[List[float]] = None, 
    y_lim: Optional[List[float]] = None, 
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    label_lists: Optional[List[List[str]]] = None, 
    linestyles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,  # New parameter for colors
    quietly: Optional[bool] = False
) -> Tuple[Figure, Axes]:
    """
    Combines multiple plots into a single plot. Each plot is given a different linestyle or color.
    
    Parameters:
    - figures_axes: A list of tuples (fig, ax), where each tuple contains a figure and its corresponding axis.
    - save_path: Optional path to save the combined plot as an image file.
    - comb_fig_title: Optional title for the combined figure.
    - x_lim: Optional list to specify x-axis limits.
    - y_lim: Optional list to specify y-axis limits.
    - label_lists: Optional list of lists of labels for the lines in each plot. If not provided, the original labels from the axes will be used.
    - linestyles: Optional list of linestyles (e.g., ['-', '--', '-.', ':']) for each plot. 
    - colors: Optional list of colors for each plot.
    - quietly: If True, suppresses the display of the plot.

    Returns:
    - A tuple (fig, ax) with the new combined figure and axis.
    """
    
    # Create a new figure and axis
    fig, ax = plt.subplots()
    # If linestyles are not provided and colors aren't provided either, set default linestyles
    if linestyles is None and colors is None:
        linestyles = ['-', '--', '-.', ':', (0, (5, 2, 2, 2))] * len(figures_axes)  # Default linestyles

    for idx, (fig_old, ax_old) in enumerate(figures_axes):
        lines = ax_old.get_lines()
        label_list = label_lists[idx] if label_lists is not None and idx < len(label_lists) else None
        # Use color if provided, otherwise fallback to linestyles
        if colors is not None and linestyles is None:
            color = colors[idx % len(colors)]
            linestyle = None  # Ignore linestyles if colors are provided
        elif colors is None and linestyles is not None:
            color = None  # Use the original color if no new colors are provided
            linestyle = linestyles[idx % len(linestyles)]
        else:
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
        # Add lines from the current plot
        for i, line in enumerate(lines):
            label = label_list[i] if label_list is not None and i < len(label_list) else line.get_label()
            ax.plot(line.get_xdata(), line.get_ydata(), label=label, 
                    color=color if color else line.get_color(), 
                    linestyle=linestyle)
    
    # Set axis labels from the first figure's axis
    ax.set_xlabel(figures_axes[0][1].get_xlabel())
    ax.set_ylabel(figures_axes[0][1].get_ylabel())
    
    # Set axis limits based on the combined range
    all_xlims = [ax_old.get_xlim() for _, ax_old in figures_axes]
    all_ylims = [ax_old.get_ylim() for _, ax_old in figures_axes]
    
    x_lim = x_lim if x_lim is not None else [min(x[0] for x in all_xlims), max(x[1] for x in all_xlims)]
    y_lim = y_lim if y_lim is not None else [min(y[0] for y in all_ylims), max(y[1] for y in all_ylims)]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    x_label = x_label if x_label is not None else figures_axes[0][1].get_xlabel()
    y_label = y_label if y_label is not None else figures_axes[0][1].get_ylabel()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend()

    # Set the title if provided
    if comb_fig_title is not None:
        ax.set_title(comb_fig_title)
    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Force canvas to draw
    fig.canvas.draw()
    fig.canvas.flush_events()   # plt.close(fig)  # Close the figure to free up memory
    if not quietly:
        plt.show()

    return fig, ax


def save_plot(figure: Figure, path: str) -> None:
    """
    Save a plot to a specified path and create the directory if it doesnt exist.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    figure.savefig(path)
    plt.close(figure)

@njit
def tuple_product(shape_tuple: Tuple[float]) -> float:
    """
    manually multiply elements in a tuple in numba
    """
    product = 1
    for dim in shape_tuple:
        product *= dim
    return product

@njit
def compute_weighted_mean(values: np.ndarray, weights: float) -> float:
    return np.dot(weights, values)

@njit
def compute_weighted_std(values: np.ndarray, weights: float, weighted_mean: float) -> float:
    weighted_var = np.dot(weights, (values - weighted_mean) ** 2)
    return np.sqrt(weighted_var)

@njit
def mean_and_sd_objective(weights: np.ndarray, values: np.ndarray, target_mean: float, target_std: float)-> float:
    """
    returns the squared difference between the weighted mean and weighted standard deviation and the target mean and standard deviation
    given weights, values, target mean, and target standard deviation
    """
    weighted_mean = compute_weighted_mean(values, weights)
    weighted_std = compute_weighted_std(values, weights, weighted_mean)
    mean_penalty = (weighted_mean - target_mean) ** 2
    std_penalty = (weighted_std - target_std) ** 2
    return mean_penalty + std_penalty

@njit
def weighted_stats(fe_weights: np.ndarray, values: np.ndarray, wage_hist: np.ndarray, sim_draws: int, H_type_weights: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the weighted mean and weighted standard deviation. 
    The weighting accounts for sim_draw_weights, H_type_weights, and fe_weights
    """
    weights = fe_weights
    weighted_mean = np.sum(weights * values)
    
    deviations =  wage_hist - weighted_mean
    squared_deviations = deviations ** 2
    weighted_squared_deviations = squared_deviations * (1.0 / sim_draws)
    weighted_squared_deviations = weighted_squared_deviations * H_type_weights[np.newaxis, :, np.newaxis]
    weighted_squared_variance = np.sum(weighted_squared_deviations * fe_weights[:, np.newaxis, np.newaxis])
    weighted_sd = np.sqrt(weighted_squared_variance)   # Calculate the weighted standard deviation

    return weighted_mean, weighted_sd

@njit
def objective(fe_weights: np.ndarray, values: np.ndarray, target_mean: float, target_std: float,
               wage_hist: np.ndarray, sim_draws: int, H_type_weights: np.ndarray) -> float:
    """
    Objective function to minimize: squared difference from target mean and std.
    """
    weighted_mean, weighted_std = weighted_stats(fe_weights, values, wage_hist, sim_draws, H_type_weights)
    mean_diff = (weighted_mean - target_mean) ** 2
    std_diff = (weighted_std - target_std) ** 2
    total_diff = mean_diff + std_diff
    return total_diff

def optimize_weights(values: np.ndarray, target_mean: float, target_std: float,
                      wage_hist: np.ndarray, sim_draws: int, H_type_weights: np.ndarray) -> np.ndarray:
    """
    Optimize the  weights to match the target mean and std as closely as possible.
    takes the following arguments:
    values: the values to be weighted
    target_mean: the target mean
    target_std: the target standard deviation
    wage_hist: the wage history
    sim_draws: the number of simulation draws
    H_type_weights: the H type weights
    returns the optimized weights
    """
    np.random.seed(42)  # Set a fixed seed for consistency
    n = len(values)
    initial_weights = np.ones(n) / n
    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n)]
    
    # Optimization using SLSQP
    result = minimize(
        objective,
        initial_weights,
        args=(values, target_mean, target_std, wage_hist, sim_draws, H_type_weights),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
    )
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    return result.x

def list_to_tex(path: str, new_tex_file_name: str, list_of_tex_lines: List[str])->None:
    """
    generates a tex file from a list of tex lines at the specified path wit the specified name
    name must include the .tex extension
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    fullpath = os.path.join(path, new_tex_file_name)
    with open(fullpath, 'w', newline='\n') as pen:
        for row in list_of_tex_lines:
            pen.write(row)
    
def tex_to_pdf(path: str, tex_file_name: str) -> None:
    """
    compiles a tex file to a pdf from the specified path with the specified name for the tex file
    pdf name is the same as the tex file name but with a pdf extension
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    fullpath = os.path.join(path, tex_file_name)
    
    # Ensure pdflatex is in the system path
    try:
        result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise EnvironmentError("pdflatex is not installed or not in the system path.")
    except FileNotFoundError:
        raise EnvironmentError("pdflatex is not installed or not in the system path.")
    
    # Compile the .tex file to a PDF
    result = subprocess.run(['pdflatex', '-output-directory', path, fullpath], capture_output=True, text=True)
    
    # Check for errors and print the output for debugging
    if result.returncode != 0:
        print(f"Error in pdflatex execution: {result.stderr}")
        print(f"Standard Output: {result.stdout}")
        raise RuntimeError(f"pdflatex failed to compile {tex_file_name}. See output above for details.")
    else:
        print(f"PDF successfully created at {os.path.join(path, tex_file_name.replace('.tex', '.pdf'))}")
    

def read_specific_column_from_csv(file_path: str, column_index: int, row_index: int = 1)-> np.ndarray:
    """
    read a specific column from a csv file skipping the header row by default
    column index is zero based 
    """
    column_values = []
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for i in range(row_index):
            next(csv_reader)
        for row in csv_reader:
            if len(row) > column_index:
                column_values.append(float(row[column_index]))  # Assuming the values are float numbers
    return np.array(column_values)

def read_specific_row_from_csv(file_path: str, row_index: int, skip_header: bool = True)-> np.ndarray:
    """
    read a specific row from a csv file skipping the header row by default
    
    """
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        # Skip the header row
        if skip_header:
            next(csv_reader)
        for i, row in enumerate(csv_reader):
            if i == row_index:
                return np.array([float(value) for value in row])  # Assuming the values are float numbers
    raise ValueError(f"Row index {row_index} is out of bounds for the file {file_path}")

def read_matrix_from_csv(file_path: str, column_index: int = 1, skip_header: bool = True) -> np.ndarray:
    """
    read a matrix from a csv file skipping the header row by default and starting at the specified column index = 1 by default
    recall that column index is zero based
    """
     # Load the entire file first to determine the number of columns
    sample_data = np.genfromtxt(file_path, delimiter=',', max_rows=1, skip_header=1 if skip_header else 0)
    num_columns = sample_data.shape[0]

    # Then read the file, specifying the correct range for usecols
    matrix = np.genfromtxt(
        file_path,
        delimiter=',',
        skip_header=1 if skip_header else 0,
        usecols=range(column_index, num_columns)  # Use the determined number of columns
    )
    return matrix

@njit
def gen_even_row_weights(matrix: np.ndarray) -> np.ndarray:
    """
    generates even weights for each row of a matrix
    """
    return np.ones(matrix.shape[0]) / matrix.shape[0]

#
def bisection_search(func: Callable, min_val: float, max_val: float, tol: float, max_iter: int, print_screen: int = 3) -> float:
    """
    function that searches for the zero of a function given a range of possible values, a function to evaluate, a tolerance, max number of iterations, and an initial guess
    this is a simple bisection method but does this take advantage of the monotoniciy of the function to speed up the search?
    """
    x0 = min_val
    x1 = max_val
    f0 = func(x0)
    f1 = func(x1)
    
    if f0 * f1 >= 0:
        raise ValueError("Function values at the endpoints have the same sign. Bisection method cannot be applied.")
    
    for i in range(max_iter):
        x_mid = (x0 + x1) / 2
        f_mid = func(x_mid)
        if print_screen >= 1:
            print(f"iteration {i}: x_mid = {x_mid}, f_mid = {f_mid}")
        
        if abs(f_mid) < tol:
            return x_mid
        
        if f_mid * f0 < 0:
            x1 = x_mid
            f1 = f_mid
        else:
            x0 = x_mid
            f0 = f_mid
    
    print("Bisection method did not converge within the specified number of iterations.")
    return x_mid
def my_bisection_search(func: callable, min_val: float, max_val: float, tol: float, max_iter: int, print_screen: int = 3) -> float:
    """
    function that searches for the zero of a function given a range of possible values, a function to evaluate, a tolerance, max number of iterations, and an initial guess
    this bisection search takes advantage of the monotoniciy of the function to speed up the search
   """ 
    low_end_point = min_val
    high_end_point = max_val
    low_end_val = func(low_end_point)
    high_end_val = func(high_end_point)

    if low_end_val * high_end_val >= 0:
        raise ValueError("Function values at the endpoints have the same sign. Bisection method cannot be applied.")
    
    for i in range(max_iter):
        mid_point = (low_end_point + high_end_point) / 2
        mid_val = func(mid_point)
        if print_screen >= 1:
            print(f"iteration {i}: mid_point = {mid_point}, mid_val = {mid_val}")
        
        if abs(mid_val) < tol:
            return mid_point
        
        if mid_val * low_end_val < 0:
            high_end_point = mid_point
            high_end_val = mid_val
        else:
            low_end_point = mid_point
            low_end_val = mid_val

    print("Bisection method did not converge within the specified number of iterations.")
    return mid_point







@njit
def create_increasing_array(shape: Tuple[int], increase_by: int = 1) -> np.ndarray:
    """
    creates an array of increasing values from 1 to the total number of elements in the array
    increasing by increase_by  each time increase_by is 1 by default
    """
    tot_elem = 1
    for dim in shape:
        tot_elem *= dim
    values = np.arange(1, tot_elem + increase_by)
    array = values.reshape(shape)
    return array

@njit(parallel=True)
def manual_kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    calculates the kronecker product of two matrices in numba
    """
    m, n = a.shape
    p, q = b.shape
    result = np.zeros((m * p, n * q), dtype=a.dtype)
    for i in prange(m):
        for j in prange(n):
            for k in prange(p):
                for l in prange(q):
                    result[i * p + k, j * q + l] = a[i, j] * b[k, l]
    return result

@njit
def gen_flat_joint_trans(trans1: np.ndarray, trans2: np.ndarray) -> np.ndarray:
    """
    generates a flattened joint transition matrix from two transition matrices
    by first calculating the kronecker product of the two matrices and then flattening the result
    """
    joint_transition = manual_kron(trans1, trans2)
    return joint_transition.flatten()

def print_exec_time (message: str , start_time: float)-> None:    
    """
    prints the execution time and a message of a function given the start time and a message
    """
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(message, execution_time, "seconds")

def rouwenhorst(N: int, rho: float, sigma: float, mu: float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rouwenhorst method to discretize AR(1) process
    The Rouwenhorst method, developed by Koen Rouwenhorst in 1995, is another technique for discretizing AR(1) processes, especially useful for highly persistent processes.
    Deals with persistency better than tauchen

    """
    q = (rho + 1)/2
    nu = ((N-1.0)/(1.0-rho**2))**(1/2)*sigma
    s = np.linspace(mu/(1.0-rho)-nu, mu/(1.0-rho)+nu, N) # states
    # implement recursively transition matrix
    P = np.array([[q, 1-q], [1-q, q]])
    for i in range(2,N): # add one row/column one by one
        P = (
            q*np.r_[np.c_[P, np.zeros(i)] , [np.zeros(i+1)]] + (1-q)*np.r_[np.c_[np.zeros(i), P] , 
                [np.zeros(i+1)]] + (1-q)*np.r_[[np.zeros(i+1)] ,  np.c_[P, np.zeros(i)]] + q*np.r_[[np.zeros(i+1)] ,  np.c_[np.zeros(i), P]]
                )
        
        P[1:i,:]=P[1:i,:]/2  # adjust the middle rows of the matrix

    return s, P

@njit
def rouwenhorst_numba(N: int, rho: float, sigma: float, mu: float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    q = (rho + 1) / 2
    nu = np.sqrt((N - 1) / (1 - rho**2)) * sigma
    s = np.linspace(mu / (1 - rho) - nu, mu / (1 - rho) + nu, N)
    P = np.array([[q, 1 - q], [1 - q, q]])

    for i in range(2, N):
        P_new = np.zeros((i + 1, i + 1))
        P_new[:i, :i] += q * P
        P_new[:i, 1:i+1] += (1 - q) * P
        P_new[1:i+1, :i] += (1 - q) * P
        P_new[1:i+1, 1:i+1] += q * P
        P = P_new
        P[1:i, :] /= 2

    return s, P

@njit
def gen_grid(size: int, min: float, max: float, grid_growth: float = 0.0) -> np.ndarray:
    """
    returns a grid of size size with min and max values and grid growth rate grid_growth
    """
    if grid_growth == 0.0:
        gA = np.linspace(min, max, size)
    elif grid_growth > 0.0:
        gA = np.zeros(size)
        for i in range(size):
            gA[i]= min + (max - (min))*((1 + grid_growth)**i -1)/((1 + grid_growth)**(size-1.0) -1)
    return gA

# I took this function from state-space Jacobian package
#because these functions are only really called internally and do not reference the instance parameters
#i do not think i need to pass in self here i could be wrong though. keep this in mind if it throws an instance or type error
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq): 
    """Efficient linear interpolation exploiting monotonicity.
    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.
    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points
    Returns
    ----------
    yq : array (nq), interpolated points
    """
    #get number of elements in xquery and x
    nxq, nx = xq.shape[0], x.shape[0]
    xi = 0 #index or ocunter maybe?
    x_low = x[0]
    x_high = x[1]

    #iterate through each query point
    for xqi_cur in range(nxq):
        # grab the current query point 
        xq_cur = xq[xqi_cur] 
        while xi < nx - 2: 
            #if the upper bound is greater than the current query point break and get new query point
            if x_high >= xq_cur: 
                break
            xi += 1 #inc counter
            x_low = x_high #set the lower endpoint to the upper one
            x_high = x[xi + 1] #set the upper endpoint to the next one in the array
        xqpi_cur = (x_high - xq_cur) / (x_high - x_low) #weight the end points 
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1] 

@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]
    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points
    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi
        xqpi[xqi_cur] =  min(max(xqpi[xqi_cur],0.0), 1.0) # if the weight is outside 0 or 1, this will catch it



####everything below comes from Blandin
# i feel like these dimension index transformations should be renamed
# since D2toD1 is returns 2 coordinates it should really be called 1Dto2D i think
@njit
def D2toD1(n, N0, N1):
    """
    Infer 2D coordinates [n0, n1] from 1D coordinate n.
    """
    n0 = n // N1
    n1 = n - n0 * N1

    return n0, n1

@njit
def D3toD1(n, N0, N1, N2):
    """
    Infer 3D coordinates [n0, n1, n2] from 1D coordinate n.
    """
    n0 = n // (N1 * N2)
    n1 = (n - n0 * N1 * N2) // N2
    n2 = n - n0 * N1 * N2 - n1 * N2

    return n0, n1, n2

@njit
def D6toD1(n, N0, N1, N2, N3, N4, N5):
    """
    Infer 6D coordinates [n0, n1, n2, n3] from 1D coordinate n.
    """
    n0 = n // (N1 * N2 * N3 * N4 * N5)
    n1 = (n - n0 * N1 * N2 * N3 * N4 * N5) // (N2 * N3 * N4 * N5)
    n2 = (n - n0 * N1 * N2 * N3 * N4 * N5 - n1 * N2 * N3 * N4 * N5) // (N3 * N4 * N5)
    n3 = (n - n0 * N1 * N2 * N3 * N4 * N5 - n1 * N2 * N3 * N4 * N5 - n2 * N3 * N4 * N5) // (N4 * N5)
    n4 = (n - n0 * N1 * N2 * N3 * N4 * N5 - n1 * N2 * N3 * N4 * N5 - n2 * N3 * N4 * N5 - n3 * N4 * N5) // N5
    n5 = (n - n0 * N1 * N2 * N3 * N4 * N5 - n1 * N2 * N3 * N4 * N5 - n2 * N3 * N4 * N5 - n3 * N4 * N5 - n4 * N5)

    return n0, n1, n2, n3, n4, n5

@njit
def D5toD1(n, N0, N1, N2, N3, N4):
    """
    Infer 5D coordinates [n0, n1, n2, n3] from 1D coordinate n.
    """
    n0 = n // (N1 * N2 * N3 * N4)
    n1 = (n - n0 * N1 * N2 * N3 * N4) // (N2 * N3 * N4)
    n2 = (n - n0 * N1 * N2 * N3 * N4 - n1 * N2 * N3 * N4) // (N3 * N4)
    n3 = (n - n0 * N1 * N2 * N3 * N4 - n1 * N2 * N3 * N4 - n2 * N3 * N4) // N4
    n4 = (n - n0 * N1 * N2 * N3 * N4 - n1 * N2 * N3 * N4 - n2 * N3 * N4 - n3 * N4)

    return n0, n1, n2, n3, n4

@njit
def D4toD1(n, N0, N1, N2, N3):
    """
    Infer 4D coordinates [n0, n1, n2, n3] from 1D coordinate n.
    """
    n0 = n // (N1 * N2 * N3)
    n1 = (n - n0 * N1 * N2 * N3) // (N2 * N3)
    n2 = (n - n0 * N1 * N2 * N3 - n1 * N2 * N3) // N3
    n3 = (n - n0 * N1 * N2 * N3 - n1 * N2 * N3 - n2 * N3)

    return n0, n1, n2, n3


@njit
def avg_wgt(v, w):
    """
    Compute weighted average of numpy matrix.
    """
    tot, totw = 0.0, 0.0
    for vi, wi in zip(v.flat, w.flat):
        tot += vi
        totw += wi
    return tot / max(1e-6, totw)

@njit
def avg_wgt_3d(v, w):
    """
    Compute weighted average of 3d matrix.
    """
    N0, N1, N2 = len(v[:, 0, 0]), len(v[0, :, 0]), len(v[0, 0, :])
    tot, totw = 0.0, 0.0
    for i in range(N0):
        for j in range(N1):
            for k in range(N2):
                tot += v[i, j, k] * w[i, j, k]
                totw += w[i, j, k]

    return tot / max(1e-6, totw)


def Taucheniid(sigma: float, num_grid_points: int, Nsd: int=3, mean: float=0.0, state_grid: np.ndarray=np.zeros(1))->Tuple[np.ndarray, np.ndarray]:
    """
    This function uses the method of Tauchen (1986) to approximate a continuous iid Normal process.

    Normal process: ε~N(0, σ**2).

    INPUTS:  -σ: SD of innovation in AR(1) process
             -S: number of gridpoints
             -Nsd: number of SD from mean for grid to span

    OUTPUTS: -state_grid, grid of state variable s
             -probs, grid of probabilities for each state
    """
    # compute grid over state s and the half-distance between gridpoints, δ
    if len(state_grid) == 1:
        state_grid = np.linspace(mean - Nsd * sigma, mean + Nsd * sigma, num_grid_points)
    δ = (state_grid[-1] - state_grid[0]) / (num_grid_points - 1) / 2

    # compute cumulative probabilities of state_grid
    probscum = np.ones(num_grid_points)
    for s in range(num_grid_points - 1):
        probscum[s] = norm.cdf(state_grid[s] + δ, loc=mean, scale=sigma)

    # compute probabilities of state_grid
    probs = probscum
    probs[1:] = probscum[1:] - probscum[:-1]

    return probs, state_grid


def AR1_Tauchen(ρ, σ, n_s=False, n_sd=False, grid=False, max_iter=200, tol=1e-5):
    """
    this function uses the method of tauchen (1986) to ...
    ... approximate a continuous "1-time" AR(1) process
    ... by a discrete markov process.

    AR(1) process: y = ρ * x + ε, where ε~N(0, σ**2).

    INPUTS:  -ρ: persistence in AR(1) process
             -σ: SD of innovation in AR(1) process
             -n_s: # of gridpoints in markov matrix
             -n_sd: # of SD from α for grid to span

    OUTPUTS: -grid: discretized grid over state space
             -M, markov transition matrix
             -M_cum, cumulative markov matrix
    """

    # infer state grid and space between gridpoints δ
    if n_s:
        grid = np.linspace(-n_sd * σ, n_sd * σ, n_s)
    else:
        n_s = len(grid)
    δ = (grid[-1] - grid[0]) / max(1, len(grid) - 1) / 2

    # initialize cumulative markov matrix
    M_cum = np.ones([len(grid), len(grid)])

    # iterate over current state s
    for ns, s in enumerate(grid):

        # iterate over next state ss
        for nss, ss in enumerate(grid[:-1]):
            M_cum[ns, nss] = norm.cdf(ss + δ, loc=ρ * s, scale=σ)

    # infer M from M_cum
    M = np.copy(M_cum)
    M[:, 1:] = M_cum[:, 1:] - M_cum[:, :-1]

    # compute stationary distribution
    distribution = np.ones(n_s) / n_s
    i = 0
    diff = tol + 1
    while i < max_iter and diff > tol:
        stationary = distribution @ M
        diff = np.amax(np.abs(stationary - distribution))
        distribution = np.copy(stationary)
        i += 1

    if i >= max_iter:
        print('AR1_Tauchen: ERROR! stationary distr. did not converge.')

    return grid, M, M_cum, stationary


def Tauchen_popwgt(gridx, gridy, vp):
    """
    This function uses the method of Tauchen (1986) to
    assign population weights to a 2-dimensional grid over
    two states (x, y), which are joint-log-Normally distributed.

    log(x, y) ~ N(μ_x, μ_y, σ_x, σ_y, ρ_xy)

    INPUTS:  -vp=(μ_x, μ_y, σ_x, σ_y, ρ_xy): distribution parameters
             -gridx: (Nx x 1) grid for state x
             -gridy: (Ny x 1) grid for state y

    OUTPUTS: -wgt, (Nx x Ny) matrix of weights for each gridpiont (x, y)
    """
    # Unpack vp
    μ_x, μ_y, σ_x, σ_y, ρ_xy = vp[0], vp[1], vp[2], vp[3], vp[4]

    # Initialize multivariate normal using distr. parameters
    distr = mvn(mean=np.array([μ_x, μ_y]),
                cov=np.array([[σ_x ** 2, ρ_xy * σ_x * σ_y], [ρ_xy * σ_x * σ_y, σ_y ** 2]]))

    # Compute logged grids
    gridlx, gridly = np.log(gridx), np.log(gridy)

    # Compute grid cutoffs
    griddlx, griddly = np.zeros(1 + len(gridx)), np.zeros(1 + len(gridy))
    griddlx[0], griddly[0] = -999, -999
    griddlx[-1], griddly[-1] = 999, 999
    for nx in range(1, len(gridx)): griddlx[nx] = (gridlx[nx] + gridlx[nx - 1]) / 2
    for ny in range(1, len(gridy)): griddly[ny] = (gridly[ny] + gridly[ny - 1]) / 2

    # Iterate over gridpoints
    wgt = np.zeros([len(gridx), len(gridy)])
    for nlx, lx in enumerate(gridlx):
        for nly, ly in enumerate(gridly):
            # Assign weights
            phihi = distr.cdf(np.array([griddlx[nlx + 1], griddly[nly + 1]]))
            philo = distr.cdf(np.array([griddlx[nlx + 1], griddly[nly]]))
            plohi = distr.cdf(np.array([griddlx[nlx], griddly[nly + 1]]))
            plolo = distr.cdf(np.array([griddlx[nlx], griddly[nly]]))
            wgt[nlx, nly] = phihi - philo - plohi + plolo

            # Check if wgt is valid
    if np.abs(np.sum(wgt) - 1) > 1e-5:
        print('Tauchen_popwgt: Weights do not sum to one!')
    if np.amin(wgt) < -1e-5:
        print('Tauchen_popwgt: Negative weights!')

    # Clean up minor rounding errors
    wgt = np.where(wgt >= 0, wgt, 0)
    wgt = wgt / np.sum(wgt)

    return wgt


def gen_aggs_quick(sim, insamp, wgt, jbar, use_insamp=True, log=False, minm=0.0, maxm=9e9):
    """
    Generate overall mean and sd of log of sim...
    ...conditional on insamp=1 and minm < sim < maxm...
    ...using weights wgt.
    """
    # is sim within cutoffs?
    insample = insamp * np.where(sim > minm, 1, 0) * np.where(sim < maxm, 1, 0)

    # Turn wgt into wgtlc
    wgtlc = np.moveaxis(np.array([wgt for j in range(jbar)]), 0, -1)
    if use_insamp:
        wgtlc = wgtlc * insample[:, :, :, :jbar]
    # Compute aggregate moments
    sim = np.where(sim[:, :, :, :jbar] > 0, sim[:, :, :, :jbar], 1e-3)
    if log:
        sim = np.log(sim)
    mn = np.average(sim, weights=wgtlc[:, :, :, :jbar])
    sd = np.average((sim - mn) ** 2, weights=wgtlc[:, :, :, :jbar]) ** (1 / 2)

    return mn, sd


def moments_1_2_weighted(x, y=False, w=1.0, by_last=False):
    """
    computes weighted mean(x), mean(y), sd(x), sd(y), cov(x,y), corr(x,y)
    if by_last==True, computes moments separately for each value of last dimension
    """
    # infer dimensions of x and y
    axes = (0,)
    for dim in range(1, len(np.shape(x)) - by_last):
        axes += (dim,)

    # moments for x
    mn_x = np.average(x, axis=axes, weights=w)
    dev_x = x - mn_x
    sd_x = np.sqrt(np.average(dev_x ** 2, axis=axes, weights=w))

    if type(y) == bool:
        return mn_x, sd_x

    # moments for y
    mn_y = np.average(y, axis=axes, weights=w)
    dev_y = y - mn_y
    sd_y = np.sqrt(np.average(dev_y ** 2, axis=axes, weights=w))
    cov_xy = np.average(dev_x * dev_y, axis=axes, weights=w)
    cor_xy = cov_xy / (sd_x * sd_y)

    return mn_x, mn_y, sd_x, sd_y, cov_xy, cor_xy


def var_decomposition(y, w, more=False):
    """
    Var(y) = Var( E[y|x] ) + E[ Var(y|x) ]
    y: (n_i, n_x) matrix
    w: (n_i, n_x) matrix
    """
    mn, sd = moments_1_2_weighted(y, w=w)

    n_x = len(y.T)
    mn_x = np.zeros(n_x)
    sd_x = np.zeros(n_x)
    for nx in range(n_x):
        if np.sum(w[:, nx]) > 0.0:
            mn_x[nx], sd_x[nx] = moments_1_2_weighted(y[:, nx], w=w[:, nx])
    w_x = np.sum(w, axis=0)
    var_between = np.average((mn_x - mn)**2, axis=0, weights=w_x)
    var_within = np.average(sd_x**2, axis=0, weights=w_x)

    if more:
        return sd**2, var_between, var_within, mn_x, sd_x**2, w_x
    else:
        return sd**2, var_between, var_within


def ppf(x, w, listp=np.arange(0.1, 1.0, 0.1)):
    """
    computes percent point function (ppf) of a sample x
    ... with associated weights w. specifically, it:
    ... i)   sorts the arrays x and w by x values
    ... ii)  assigns cdf values to each x value based on w
    ... iii) for each p in listp, returns the first x value whose
    ... cdf value exceeds p.
    """

    # sort x and w by x values
    xsort = np.array(sorted(x))
    wsort = [w for _, w in sorted(zip(x, w))] / np.sum(w)

    # assign cdf values to each x value based on w
    wcum = np.cumsum(wsort)

    # for each p in listp, return the first x value whose cdf value exceeds p
    listppf = []
    for p in listp:
        first_exceeding_index = np.argmax(wcum >= p)
        listppf.append(xsort[first_exceeding_index])

    return listppf


def pratio(x, w, pnumer=[0.9, 0.9, 0.5, 0.75], pdenom=[0.10, 0.5, 0.1, 0.25]):
    """
    computes percentile ratios, pnum / pdenom, for array x
    :param x: sample values
    :param w: sample weights
    :param pnumer: list of numerators for percentile ratios
    :param pdenom: list of denominators for percentile ratios
    :return: percentile ratios
    """
    # get ppf of x corresponding to pnumer then corresponding to pdenom
    ppfn = np.array(ppf(x, w, listp=np.array(pnumer)))
    ppfd = np.array(ppf(x, w, listp=np.array(pdenom)))

    return ppfn / ppfd


def make_lagged_pairs(x, lag=1):
    """
    n_r rows of x are individuals, n_c columns are time periods, lag is lag length.
    return (n_r * (n_c - lag) -by- 2) array of lag pairs (by individual)
    """
    n_r = len(x)
    n_c = len(x.T)
    lagged_pairs = np.zeros([n_r * (n_c - lag), 2])
    for r in range(n_c - lag):
        index = n_r * r
        lagged_pairs[index:index + n_r, 0] = x[:, r]
        lagged_pairs[index:index + n_r, 1] = x[:, r + lag]
    return lagged_pairs


def plot_histogram(partition,
                   data1, mult1=100, lbl1='Model', ymax=65, ylabel='Share (%)',
                   data2=None, mult2=100, lbl2='Data', ymax2=None, ylabel2='Share (%)',
                   mn1=np.NaN, sd1=np.NaN, mn2=np.NaN, sd2=np.NaN,
                   xtic=1,
                   vlabel='', vlbl='', path=''):
    """
    plot deciles of m1 and m2 using weights w1, w2.
    plot alongside data deciles.
    """
    # compute shares in each partition cell
    fig, ax = plt.subplots()
    width = (partition[2] - partition[1]) / 3
    ax.bar(partition[:-1] + width / 2, mult1 * data1, width=width, color='tab:orange', label=lbl1)
    ax.set_xticks(partition[:-1:xtic])
    ax.set_ylim([0, ymax])
    ax.set_xlabel(vlabel)
    ax.set_ylabel(ylabel)
    if data2 is not None:
        if ymax2 is None:
            ax.bar(partition[:-1] - width / 2, mult2 * data2, width=width, color='tab:blue', label=lbl2)
        else:
            ax2 = ax.twinx()
            ax2.bar(partition[:-1] - width / 2, mult2 * data2, width=width, color='tab:blue', label=lbl2)
            ax2.set_ylim([0, ymax2])
            ax2.set_ylabel(ylabel2)
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2)
    if ymax2 is None:
        ax.legend()
    fig.savefig(path + f'fig_histogram_{vlbl}.pdf', bbox_inches='tight')
    plt.close()
    with open(path + f'fig_histogram_{vlbl}.csv', 'w', newline='') as f:
        pen = csv.writer(f)
        pen.writerow(['vlabel', 'mean', 'sd'] + list(partition[:-1]))
        pen.writerow([lbl1, mn1, sd1] + list(mult1 * data1))
        if data2 is not None:
            pen.writerow([lbl2, mn2, sd2] + list(mult2 * data2))

@njit
def quadratic(j: int, params: np.ndarray) -> float:
    """
    evaluate quadratic in j given parameters params = [p0, p1, p2]
    """
    return params[0] + params[1] * j + params[2] * j ** 2

@njit
def cubic(j : int, params : np.ndarray) -> float:
    """
    evaluate cubic in j given parameters params = [p0, p1, p2, p3]
    """
    return params[0] + params[1] * j + params[2] * j ** 2 + params[3] * j ** 3

#run if main
if __name__ == "__main__":
    print("running my_toolbox.py")
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    input_path = main_path + "input/"
    file_path = input_path + "MH_trans.csv"

    print("done running my_toolbox.py")