clear all
cls

global curdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model"  /* main folder directory */ 
global dodir    "$curdir/My Code/MH_Model/do_files" /*do file folder directory */
global master_outdir   "$curdir/My Code/MH_Model/model_uncert/input" /*output file folder directory */
global datadir  "$curdir/data" //"C:\Users\Ben\3YP_Data_Work\Data"

// for the health transition moments
// global outdir "$master_outdir/MH_trans"

// do "$dodir/MH_trans_2_types.do" //generate the mental health transition probabilities for the model

// do "$dodir/validate_H_types.do" //validate the health types generated in the model by regression

// for the other moments
global outdir "$master_outdir/50p_age_moms"

// do "$dodir/lt_earn_stats.do"  

do "$dodir/moments_giv_H_type.do" //generate the moments for the model with given binary permanent health types

// do "$dodir/MP_health_age.do"  

// do "$dodir/earnings_var_perc.do" //generate the earnings inequality moments 

// do "$dodir/autocorr.do" //generate the autocorrelations of health and earnings

// do "$dodir/earnings_and_health.do" //generate the earnings by health moments

di "*****got to the end of main_model_moms.do*****"