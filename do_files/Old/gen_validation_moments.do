cls

clear

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files" /*do file folder directory */
global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input/k-means/" /*output file folder directory */
global datadir  "$curdir/Data"

local dir = "$datadir"

cd "`dir'"
// use "`dir'/UKHLS_sample_probs", clear
use "`dir'/UKHLS_sample_k2", clear

local start_age = 25
local end_age = 75

preserve
collapse (sd) log_labor_inc if age >= `start_age' & age <= `end_age', by(age)

rename log_labor_inc var_log_lab_inc
replace var_log_lab_inc = var_log_lab_inc * var_log_lab_inc
tempfile var_log_lab_inc_data
save `var_log_lab_inc_data', replace
restore

sort indiv_id year
bysort indiv_id (year): gen cum_lab_inc = sum(labor_income)
*list indiv_id year labor_income cum_lab_inc in 1/20
gen log_cum_lab_inc = log(cum_lab_inc)

preserve
collapse (sd) log_cum_lab_inc if age >= `start_age' & age <= `end_age', by(age)
rename log_cum_lab_inc var_log_cum_lab_inc
replace var_log_cum_lab_inc = var_log_cum_lab_inc * var_log_cum_lab_inc
tempfile var_log_cum_lab_inc_data 
save `var_log_cum_lab_inc_data', replace
restore

preserve
*Merge and save
use `var_log_lab_inc_data', clear
export delimited using "var_earn_moments.csv", replace

* Read the wages dataset into memory
import delimited "var_earn_moments.csv", clear

* Add the standard deviation to the dataset as a new observation

merge 1:1 age using `var_log_cum_lab_inc_data'
 
* Export the dataset to a CSV file
cd "$outdir"
export delimited using "var_earn_moments.csv", replace
restore

* Define the percentiles to calculate
local percentiles "5 10 50 90 95"

preserve
* Create a new dataset to store results
tempfile percentiles_data
clear
set obs 1
generate age = .
generate p5 = .
generate p10 = .
generate p50 = .
generate p90 = .
generate p95 = .
save `percentiles_data', replace
restore

* Loop over each age group to calculate the percentiles
levelsof age, local(ages)
foreach a of local ages {
    preserve
    keep if age == `a'
    
    * Check if there is data for this age
    if _N > 0 {
        summarize labor_income if emp == 1 & wage > 10, detail
        
        * Get the percentile values
        local p5 = r(p5)
        local p10 = r(p10)
        local p50 = r(p50)
        local p90 = r(p90)
        local p95 = r(p95)
        
        * Create a temporary dataset to store this result
        clear
        set obs 1
        generate age = `a'
        generate p5 = `p5'
        generate p10 = `p10'
        generate p50 = `p50'
        generate p90 = `p90'
        generate p95 = `p95'
        
        tempfile temp_data
        save `temp_data', replace
        
        * Append to the main dataset
        use `percentiles_data', clear
        append using `temp_data'
        save `percentiles_data', replace
    }
    restore
}

preserve
* Export the results to a CSV file
use `percentiles_data', clear
export delimited using "var_earn_percentiles.csv", replace
restore



