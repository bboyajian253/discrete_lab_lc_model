"""
My Model 2 - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
import pars_shocks_and_wages as ps
import model_no_uncert as model
import my_toolbox as tb

main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output"

def main():
    print("Running main")
    myPars = ps.Pars(main_path)
    print("Path:", myPars.path)

#run stuff
main()