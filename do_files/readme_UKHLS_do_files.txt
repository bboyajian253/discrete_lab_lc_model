*Readme.txt for stata .do files that process UKHLS data

main_UKHLS.do
    defines global directories and runs the following .do files in order:
    global curdir "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model"
    global dodir  "$curdir/My Code/MH_Model/do_files"
    global datadir  "$curdir/data" //"C:\Users\Ben\3YP_Data_Work\Data"
    This "main" file runs the following other .do files in order:
    - UKHLS_data_clean.do
    - UKHLS_sample.do
    - UKHLS_trans.do
    - UKHLS_quants.do
    //UKHLS_labor_prod.do 
    //do $dodir/My_UKHLS_table_format_test.do

UKHLS_data_clean.do
    Cleans, combines and renames data from the original UKHLS data files.
    saves the cleaned data as UKHLS_sample_clean.dta

UKHLS_sample.do
    takes UKHLS_sample_clean.dta
    Defines 3 samples based on the process of De Nardi, Pashchenko, and Porapakkarm(2017)
    May be able to remove... theres a lot of unused code in here
    but there may be some important setup with the survey weights
    calls gen_health_states_types.do
    saves the samples as UKHLS_sample.dta

gen_health_states_types.do
    can take UKHLS_sample data but it is commented out
    insted is called inside UKHLS_sample.do
    generates the health states and types for the model
    states gen based on qunatiles q = 5 for now
    types gen based on k-means clustering k = 2 and 50th percentile for now

UKHLS_trans.do
    takes UKHLS_sample_clean.dta
    generate mental, physical, and composite health types and transition probabilities
    saves the probabilities as UKHLS_sample_trans.dta

UKHLS_quants.do
    takes UKHLS_sample_trans.dta
    Create and analyze the quantiles of the continuous mental_health and physical_health variable
    saves the quantiles as UKHLS_sample_quants.dta


UKHLS_labor_prod.do 
    takes UKHLS_sample_trans.dta
    estimate labor productivity, and income and employment profiles by health again following De Nardi, Pashchenko, and Porapakkarm(2017)
    again a lot of unused code in here
    save shocks in UKHLS_healthshocks.dta and UKHLS_laborshocks.dta - this is uncetainty to the wage process?

gen_csv_model_sample.do
    takes UKHLS_sample.dta
    exports the main sample data to a .csv file 

******Below is for the model estimation and validation

main_model_moms
    defines global directories and runs the following .do files in order:
    global curdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model"
    global dodir    "$curdir/My Code/MH_Model/do_files"
    global master_outdir   "$curdir/My Code/MH_Model/model_uncert/input"
    global datadir  "$curdir/data"
    // for the health transition moments
    global outdir "$master_outdir/MH_trans"
    do "$dodir/gen_MH_trans_2_types.do" //generate the mental health transition probabilities for the model
    do "$dodir/validate_H_types.do" //validate the health types generated in the model by regression
    // for the other moments
    global outdir "$master_outdir/k2_moms"
    do "$dodir/gen_moments_k2.do" //generate the moments for the model with 2 health types
    do "$dodir/gen_inequality_moments.do" //generate the inequality moments for the model with 2 health types

gen_MH_trans_2_types.do
    takes UKHLS_sample_trans.dta
    Generates the mental health transition probabilities for the model unconditionally, conditonal on both permanent types and age
    saves the following .csv files in 
    global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/model_uncert/input/MH_trans/" 
    -MH_trans_uncond.csv
    -MH_trans_uncond_age.csv
    -MH_trans_by_MH_clust_50p.csv
    -MH_trans_by_MH_clust_k2.csv
    -MH_trans_by_MH_clust_50p_age.csv
    -MH_trans_by_MH_clust_k2_age.csv
    saves the data with the transitions as UKHLS_sample_trans.dta

validate_H_types.do
    takes UKHLS_sample_trans.dta
    validates the health types by regressing mental health on the health types and other controls
    exports the resulting R^2 to MH_type_reg_r2.csv
    saves the following .csv files in 
    global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/model_uncert/input/MH_trans/"
    -MH_type_reg_r2.csv
    doesnt save any .dta files

gen_moments_k2.do
    takes UKHLS_sample_trans.dta
    gens aggregate moments for model calibration and graphs 
    gens mental health transitions for the model using k-means clustering unconditonally and by age
    saves the following .csv files in 
    global outdir "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/model_uncert/input/k-means/" 
    -"MH_wage_moments.csv"
    -labor_moments.csv
    -wage_moments.csv
    -emp_rate_moments.csv
    -MH_clust_k2_pop_shares.csv
        # ***** may want to change how trans is generated its redundant in do file.
    -MH_trans_uncond.csv
    -MH_trans_by_MH_clust.csv
    -MH_trans_by_MH_clust_age.csv
    saves the data as UKHLS_sample_k2.dta

gen_inequality_moments.do
    takes UKHLS_sample_k2.dta
    gen variance and percentile and inequality moments for model validation
    gens mental health and earnings correlation moments
    saves the following .csv files in 
    global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/model_uncert/input/k-means/"
    -var_earn_moments.csv
    -earn_percentiles.csv
    -cum_earn_percentiles.csv
    -cont_MH_earn_corr.csv
    -MH_earn_corr.csv
    doesnt save any .dta files
