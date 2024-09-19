
clear all
cls

// set maxvar 20000

global curdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model"  /* main folder directory */ 
global dodir    "$curdir/My Code/MH_Model/do_files" /*do file folder directory */
global master_outdir   "$curdir/My Code/MH_Model/my_code/model_uncert/input" /*output file folder directory */
global datadir  "$curdir/data" //"C:\Users\Ben\3YP_Data_Work\Data"

// for the health transition moments
global outdir "$master_outdir/MH_trans"
do "$dodir/gen_MH_trans_2_types.do" //generate the mental health transition probabilities for the model

do "$dodir/validate_H_types.do" //validate the health types generated in the model by regression

// for the other moments
global outdir "$master_outdir/k2_moms"
do "$dodir/gen_moments_k2.do" //generate the moments for the model with 2 health types

global outdir "$master_outdir/k2_moms"
do "$dodir/gen_inequality_moments.do" //generate the inequality moments for the model with 2 health types
di "*****got to the end of main_model_moms.do*****"