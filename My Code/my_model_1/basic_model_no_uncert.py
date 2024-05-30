"""
My Model 1 - Basic Model

This script implements a basic model for the My Model 1 project.

Author: Ben Boyajian
Date: 2024-05-29 20:16:01
"""

import time
import numpy as np
from pars_shocks_and_wages import Pars
import my_toolbox
from math import exp,sqrt,log
from numba import njit, guvectorize, prange 
from interpolation import interp

@njit
def util(myPars : Pars, c, leis) :
    """
    utility function
    """
    sig = myPars.sigma
    alpha = myPars.alpha
    return (1/(1-sig)) * (c**alpha) * (leis**(1-alpha))**(1-sig)

@njit
def leis_giv_c(myPars : Pars, c, wage) :
    """
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for leisure given current period consumption
    """
    constant = (myPars.phi_n * (1 - myPars.alpha)) / (wage * myPars.alpha)
    return constant * c

@njit
def c_giv_leis(myPars : Pars, leis, wage) :
    """
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for consumption given current period leisure
    """
    constant = (wage * myPars.alpha) / (myPars.phi_n * (1 - myPars.alpha))
    return constant * leis



if __name__ == "__main__":
    print("Hello, World!")
