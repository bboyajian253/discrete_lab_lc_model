"""
Created on 2024-05-13 09:49:08

@author: Ben Boyaian
"""
#import stuff
import time
import pars_and_shocks
import numpy as np
import my_toolbox
from math import exp,sqrt,log
from numba import njit, guvectorize, prange # NUMBA speed up quite a lot, see the functions that have the decorator just above
# this is the meat of the exercise
#from scipy.optimize import brentq  # root-finding routine probably dont need since not solving to convergence

######Define the class and intialize some stuff
class my_toy_ls_model() :
    def __init__(self, pars: pars_and_shocks) :
        ###initialize earnings parameters###
        # lambda_t_H deterministic component
        # constant and age components
        self.pars = pars
        self.w_determ_cons,self.w_age,self.w_age_2,self.w_age_3 = pars.w_determ_cons,pars.w_age,pars.w_age_2,pars.w_age_3
        # health components
        self.w_avg_good_health,self.w_avg_good_health_age,self.w_good_health,self.w_good_health_age = pars.w_avg_good_health,pars.w_avg_good_health_age,pars.w_good_health,pars.w_good_health_age
        # nu_t persistent AR(1) shock
        self.rho_nu, self.sigma_eps_2, self.sigma_nu0_2 = pars.rho_nu, pars.sigma_eps_2, pars.sigma_nu0_2
        self.sigma_eps = sqrt(self.sigma_eps_2) #convert from variance to standard deviations
        self.nu_grid_size = pars.nu_grid_size
        self.nu_grid,self.nu_trans = my_toolbox.rouwenhorst(self.nu_grid_size, self.rho_nu, self.sigma_eps)
        # gamma fixed productiviy drawn at birth
        self.sigma_gamma_2 = pars.sigma_gamma_2
        self.lab_FE_list = pars.lab_FE_list

        ###iniatlize utlity parameters###
        self.beta,self.alpha,self.sigma_util = pars.beta,pars.alpha,pars.sigma_util
        #iniatlize health and time cost parameters
        self.phi_n,self.phi_H = pars.phi_n,pars.phi_H
        #health transition probabilities toy examples
        self.B2B,self.G2G = pars.B2B,pars.G2G

        ###interest rate and maybe taxes later
        self.r = pars.r

        ###intialie grid(s)
        ###define asset grid###
        self.a_min,self.a_max = pars.a_min,pars.a_max
        self.a_grid_size = pars.a_grid_size
        self.a_grid_growth = pars.a_grid_growth
        self.a_grid = my_toolbox.gen_grid(self.a_grid_size, self.a_min, self.a_max, self.a_grid_growth)

        ###initialize time/age, number of draws and other procedural parameters
        self.dt,self.dtdraws,self.T = pars.dt,pars.dtdraws,pars.T
        self.print_screen = pars.print_screen

        #value function for all states; 0=age/time, 1=assets, 2=health, 3=fixed effect
        #self._VF = np.zeros((self.nt+1,self.a_grid_size,2,2))

        #define a dictionary containing all params
        self.param_dict = {
                key:value for key, value in self.__dict__.items()
                if not key.startswith('__')
                and not callable(value)
                and not callable(getattr(value, "__get__", None)) # <- important
            }

        # if class is this class, then do some stuff
        if self.__class__.__name__ == 'my_toy_ls_model' :
            if self.print_screen >= 2 :
                print("This is a toy labor supply lifecycle model")
                print('The PARAMETERS are as follows:')
                print()
                print(self.param_dict)

 ######define some other helper functions




#####define the pieces of the HH problem
    #return leisure given labor choice and health status
    @njit
    def leisure_giv(myPars, labor, health):
        leisure = 1 - labor*myPars.phi_n -health*myPars.phi_H
        return leisure
    # the utility function given a consumption and labor choice and a current health status
    @njit
    def util(myPars, consum, labor, health) :
        leisure = myPars.leisure_giv(labor, health)
        util = ((consum**(myPars.alpha)*leisure**(1-myPars.alpha))**(1-myPars.sigma_util))/(1-myPars.sigma_util)
        #print("VALID CHOICES: ", consumption_choice, " and ", leisure_choice, " util is: ", util)
        return util
    #derivative of the util function wrt consumption c
    @njit
    def util_c(myPars, c, labor, health):
        #assign/unpack variable and param values
        l = myPars.leisure_giv(labor,health)
        alpha,sigma=myPars.alpha,myPars.sigma_util
        #evaluate the derivative determined analytically
        eval = alpha*c**(alpha - 1)*l**(1 - alpha)/(c**alpha*l**(1 - alpha))**sigma
        return eval

    @njit
    def util_l(self, c, lab, health) :
        #assign/unpack variable and param values
        l=leisure_giv(myPars, lab, health)
        a,s=self.alpha,self.sigma_util
        #evaluate the derivative determined  analytically
        eval = c**a*(1 - a)/(l**a*(c**a*l**(1 - a))**s)
        return eval

    # return the orpimal consumption decision given an asset choice a_prime and a current asset level, health statues, and labor income
    @njit
    def c_star(self, a_prime, a, health, lab_inc, ) :
        consum = self.alpha*((lab_inc/self.phi_n)*(1 - self.phi_H*health)+(1 + self.r)*a - a_prime)
        return consum

    # return the optimal labor decision given an asset choice a_prime and a current asset level, health statues, and labor income
    @njit
    def n_star(self, a_prime, a, health, lab_inc) :
        lab =  ( (self.alpha/self.phi_n)*(1 - self.phi_H*health)
                +((self.alpha - 1)/lab_inc)((1 + self.r)*a - a_prime)
                )
        return lab
    
    # return the orpimal leisure decision given an asset choice a_prime and a current asset level, health statues, and labor income
    @njit
    def l_star(self, a_prime, a, health, lab_inc) :
        leis = self.leisure_giv(self.n_star(a_prime, a, health, lab_inc), health)
        return leis
    
    @njit
    def det_lab_inc(age, health, myPars) :
        """returns the deterministic part of the wage"""
        age_comp = myPars.w_determ_cons + myPars.w_age*age + myPars.w_age_2*age*age + myPars.w_age_3*age*age*age 
        #gonna ignore average health which is in Capatina and the iniatialization of theis program for now, will work in more health states when i have composite types working.
        health_comp = myPars.w_good_health*(1-health) + myPars.w_good_health_age*age*(1-health)
        comp = age_comp + health_comp
        #print("Comp is ", comp) 
        return comp
    
    # @njit
    # def lab_inc() :
    #     pass
    
if __name__ == "__main__":
    import pars_and_shocks
    print("Running main")
    start_time = time.time()
    
    myPars = pars_and_shocks.pars()
    myModel = my_toy_ls_model(myPars)
    for age in prange(myModel.T):
        health = 0
        print('Age: ', age)
        print('Health:', health)
        det_lab_inc = my_toy_ls_model.det_lab_inc(age, health, myPars)
        print('For age ', age, ' and health ', health, ' the det lab inc is: ', det_lab_inc)

    #print(myModel.param_dict) 

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
