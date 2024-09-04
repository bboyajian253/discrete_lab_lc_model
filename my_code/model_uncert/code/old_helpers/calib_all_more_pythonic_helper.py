import numpy as np
from typing import Tuple, Dict

def calib_all(myPars: Pars, calib_path: str, alpha_mom_targ: float,  w0_mean_targ: float, w0_sd_targ: float, w1_mom_targ: float, w2_mom_targ: float, wH_mom_targ: float,
              w1_min: float = 0.0, w1_max: float = 10.0, w2_min = -1.0, w2_max = 0.0, wH_min = -5.0, wH_max = 5.0, wH_tol: float = 0.001,
              alpha_tol: float = 0.001, w0_mom_tol: float = 0.001, w1_tol: float = 0.001, w2_tol: float = 0.001) -> Tuple[float, np.ndarray, float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Calibrates all the parameters of the model.
    Takes arguments that represent the targets and tolerances for the calibration.
    Returns a tuple with the calibrated parameters, the state solutions, and the simulations.
    """
    # Set up return arrays
    state_sols = {}
    sims = {}
    my_alpha_moment = -999.999
    my_w0_mean_mom = -999.999
    my_w0_sd_mom = -999.999
    my_w1_moment = -999.999
    my_w2_moment = -999.999
    my_wH_moment = -999.999

    def check_moments():
        my_w0_mean_mom, my_w0_sd_mom = w0_moments(myPars)
        my_w1_moment = w1_moment(myPars)
        my_w2_moment = w2_moment(myPars)
        my_wH_moment = wH_moment(myPars)
        return my_w0_mean_mom, my_w0_sd_mom, my_w1_moment, my_w2_moment, my_wH_moment

    calibrations = [
        ("w0", calib_w0, w0_mom_tol, (w0_mean_targ, w0_sd_targ)),
        ("w1", calib_w1, w1_tol, (w1_mom_targ, None, w1_min, w1_max)),
        ("w2", calib_w2, w2_tol, (w2_mom_targ, None, w2_min, w2_max)),
        ("wH", calib_wH, wH_tol, (wH_mom_targ, None, wH_min, wH_max)),
        ("alpha", calib_alpha, alpha_tol, (alpha_mom_targ,))
    ]

    for i in range(myPars.max_calib_iters):
        print(f"Calibration iteration {i}")
        for name, calib_func, tol, targets in calibrations:
            print(f"***Calibrating {name}***")
            calib_output = calib_func(myPars, calib_path, tol, *targets)
            
            if name == "w0":
                w0_weights, my_w0_mean_mom, my_w0_sd_mom, state_sols, sims = calib_output
            else:
                calib_param, my_moment, state_sols, sims = calib_output
            
            my_w0_mean_mom, my_w0_sd_mom, my_w1_moment, my_w2_moment, my_wH_moment = check_moments()
            
            if (np.abs(my_w0_mean_mom - w0_mean_targ) + np.abs(my_w0_sd_mom - w0_sd_targ) < w0_mom_tol
                and (name != "w1" or np.abs(my_w1_moment - w1_mom_targ) < w1_tol)
                and (name != "w2" or np.abs(my_w2_moment - w2_mom_targ) < w2_tol)
                and (name != "wH" or np.abs(my_wH_moment - wH_mom_targ) < wH_tol)
                and (name != "alpha" or np.abs(my_alpha_moment - alpha_mom_targ) < alpha_tol)):
                print(f"***{name} calibrated***")
            else:
                print(f"***{name} calibration failed***")
                break
        else:
            # If we don't break, all parameters are calibrated
            print(f"Calibration converged after {i+1} iterations")
            print(f"w0_weights = {w0_weights}, w0_mean = {my_w0_mean_mom}, w0_mean_targ = {w0_mean_targ}, w0_sd = {my_w0_sd_mom}, w0_sd_targ = {w0_sd_targ}")
            print(f"w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_moment}, w1 mom targ = {w1_mom_targ}")
            print(f"w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_moment}, w2 mom targ = {w2_mom_targ}")
            print(f"wH = {myPars.wH_coeff}, wH moment = {my_wH_moment}, wH mom targ = {wH_mom_targ}")
            print(f"alpha = {myPars.alpha}, alpha moment = {my_alpha_moment}, alpha mom targ = {alpha_mom_targ}")
            return myPars.alpha, myPars.lab_FE_weights, myPars.wage_coeff_grid[1,1], myPars.wage_coeff_grid[1,2], myPars.wH_coeff, state_sols, sims

    # If calibration does not converge
    print(f"Calibration did not converge after {myPars.max_calib_iters} iterations")
    print(f"w0_weights = {w0_weights}, w0_mean = {my_w0_mean_mom}, w0_mean_targ = {w0_mean_targ}, w0_sd = {my_w0_sd_mom}, w0_sd_targ = {w0_sd_targ}")
    print(f"w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_moment}, w1 mom targ = {w1_mom_targ}")
    print(f"w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_moment}, w2 mom targ = {w2_mom_targ}")
    print(f"wH = {myPars.wH_coeff}, wH moment = {my_wH_moment}, wH mom targ = {wH_mom_targ}")
    print(f"alpha = {myPars.alpha}, alpha moment = {my_alpha_moment}, alpha mom targ = {alpha_mom_targ}")
    return myPars.alpha, myPars.lab_FE_weights, myPars.wage_coeff_grid[1,1], myPars.wage_coeff_grid[1,2], myPars.wH_coeff, state_sols, sims
