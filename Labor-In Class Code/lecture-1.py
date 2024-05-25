#%%
'''
In-class test: extend the model to allow for probability of divorce or marriage
'''

from ddcpmodel import ddcpmodel
import numpy as np

class ddcp_marriage(ddcpmodel):
    # class for the model with marriage dynamics
    # 2 additional parameters:
    # probability of staying married if married: p_stay_married
    # probability of getting married if unmarried: p_get_married

    def __init__(self, 
                 p_stay_married = .8, # probability of staying married
                 p_get_married = .5,  # probability of getting married
                 *args, **kwargs):
        
        self.p_stay_married = p_stay_married
        self.p_get_married = p_get_married
        
        # initialize the parent class to load the other parameters
        ddcpmodel.__init__(self, *args, **kwargs)
        

        
        # if class is ddcp_marriage, then compute the value function 
        if self.__class__.__name__ == 'ddcp_marriage' :
            self.ValueFunction()

        return
    
    ''' think about what function(s) from ddcp models you need to change'''

def test_equality_of_VF(mod1, mod2):

    return np.sum(mod1._VF != mod2._VF)/np.size(mod1._VF)         

def main():

    mod1 = ddcpmodel()
    mod2 = ddcp_marriage(p_stay_married=1, p_get_married=0, w_educ=[0,1,2])
    check = test_equality_of_VF(mod1, mod2)
    print('fraction incorrect:', check)

    return mod1, mod2

#%%
if __name__ == '__main__':
    models = main()

# %%
