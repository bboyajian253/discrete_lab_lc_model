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
import time
from typing import List, Dict, Tuple, Callable

@njit
def gen_even_weights(matrix: np.ndarray) -> np.ndarray:
    """
    generates even weights for each row of a matrix
    """
    return np.ones(matrix.shape[0]) / matrix.shape[0]

#function that searches for the zero of a function given a range of possible values, a function to evaluate, a tolerance, max number of iterations, and an initial guess
# this is a simple bisection method but this take advantage of the monotoniciy of the function to speed up the search?
def bisection_search(func: Callable, min_val: float, max_val: float, tol: float, max_iter: int, print_screen: int = 3) -> float:
    x0 = min_val
    x1 = max_val
    f0 = func(x0)
    f1 = func(x1)
    
    if f0 * f1 >= 0:
        raise ValueError("Function values at the endpoints have the same sign. Bisection method cannot be applied.")
    
    for i in range(max_iter):
        x_mid = (x0 + x1) / 2
        f_mid = func(x_mid)
        if print_screen > 0:
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

@njit
def create_increasing_array(shape):
    # Calculate the total number of elements in the array
    tot_elem = 1
    for dim in shape:
        tot_elem *= dim
    
    # Create a 1D array with increasing values starting from 1
    values = np.arange(1, tot_elem + 1)
    
    # Reshape the array to the desired shape
    array = values.reshape(shape)
    
    return array


def write_nd_array(writer, array, depth=0):
    if array.ndim == 1:
        writer.writerow(['  ' * depth + str(element) for element in array])
    else:
        for sub_array in array:
            write_nd_array(writer, sub_array, depth + 1)
            writer.writerow([])  # Blank row for separation at each level

@njit(parallel=True)
def manual_kron(a, b):
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
def gen_flat_joint_trans(trans1, trans2):
    joint_transition = manual_kron(trans1, trans2)
    return joint_transition.flatten()

def print_exec_time ( mess : str , start_time) :    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(mess, execution_time, "seconds")

def rouwenhorst(N, rho, sigma, mu=0.0):
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
def rouwenhorst_numba(N, rho, sigma, mu=0.0):
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

### generates a grid with curvature/grid growth
@njit
def gen_grid(size, min, max, grid_growth = 0.0) :
        #### Define the asset grid:
    if grid_growth == 0.0:
        #if no grid curvature 
        gA = np.linspace(min, max, size)
    elif grid_growth > 0.0:
        #if grid curvature
        gA = np.zeros(size)
        #make empty asset grid
        for i in range(size):
            #fill each cell with a point that accounts for the curvature the grid will be denser in certain places i.e where changes in mu are highest?
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
        # while the index is less than the size of x -2 
        while xi < nx - 2: 
            #if the upper bound is greater than the current query point break and get new query point
            if x_high >= xq_cur: 
                break
            #else we have x_high < xq_curr
            xi += 1 #inc counter
            #since the the upper end point is less than the query
            x_low = x_high #set the lower endpoint to the ipper one
            #raise the value of the upper end pint
            x_high = x[xi + 1]
        #get weights for the endpoints to use in the linear interpolation
        xqpi_cur = (x_high - xq_cur) / (x_high - x_low) 
        #weight the end points and do the interpolation store the result in yq at the curr query points location (index-i)
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


def Taucheniid(std_dev: float, num_grid_points: int, Nsd: int=3, mean: float=0.0, state_grid: np.ndarray=np.zeros(1))->Tuple[np.ndarray, np.ndarray]:
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
        state_grid = np.linspace(mean - Nsd * std_dev, mean + Nsd * std_dev, num_grid_points)
    δ = (state_grid[-1] - state_grid[0]) / (num_grid_points - 1) / 2

    # compute cumulative probabilities of state_grid
    probscum = np.ones(num_grid_points)
    for s in range(num_grid_points - 1):
        probscum[s] = norm.cdf(state_grid[s] + δ, loc=mean, scale=std_dev)

    # compute probabilities of state_grid
    probs = probscum
    probs[1:] = probscum[1:] - probscum[:-1]

    return state_grid, probs


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
    print( gen_even_weights(np.ones(10)) )
    print("done running my_toolbox.py")