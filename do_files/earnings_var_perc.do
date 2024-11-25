cls

clear

local dir = "$datadir"

cd "`dir'"
// use "`dir'/UKHLS_sample_trans", clear
use "`dir'/UKHLS_sample_k2", clear

local start_age = 25
local end_age = 75

preserve
// collapse (sd) log_labor_earnings if age >= `start_age' & age <= `end_age' & emp == 1, by(age)
collapse (sd) log_labor_earnings if emp == 1 & log_labor_earnings != . & labor_earnings > 0, by(age)
rename log_labor_earnings var_log_lab_inc
replace var_log_lab_inc = var_log_lab_inc * var_log_lab_inc
tempfile var_log_lab_inc_data
save `var_log_lab_inc_data', replace
restore


cd "$outdir"

preserve
*Merge and save
use `var_log_lab_inc_data', clear
export delimited using "var_earn_moments.csv", replace
restore

* Define the percentiles to calculate
local percentiles "5 10 25 50 75 90 95"

preserve
* Create a new dataset to store results
tempfile percentiles_data
clear
set obs 1
generate age = .
generate p5 = .
generate p10 = .
generate p25 = .
generate p50 = .
generate p75 = .
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
        summarize labor_earnings if emp == 1, detail
        
        * Get the percentile values
        local p5 = r(p5)
        local p10 = r(p10)
        local p25 = r(p25)
        local p50 = r(p50)
        local p75 = r(p75)
        local p90 = r(p90)
        local p95 = r(p95)
        
        * Create a temporary dataset to store this result
        clear
        set obs 1
        generate age = `a'
        generate p5 = `p5'
        generate p10 = `p10'
        generate p25 = `p25'
        generate p50 = `p50'
        generate p75 = `p75'
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
export delimited using "earn_percentiles.csv", replace
restore


preserve
* Create an empty dataset with the variables 'age' and 'correlation'
tempfile corr_results
clear
set obs 0
gen age = .
gen correlation = .
save `corr_results', replace
restore

* Loop over each age group and calculate correlation
levelsof age, local(ages)
foreach a of local ages {
    preserve
    keep if age == `a'
    
    * Check if there's data for the current age
    if _N > 0 {
        correlate mental_health labor_earnings if emp == 1
        matrix corr_matrix = r(C)
        local corr = corr_matrix[1,2]

        * Append this age's result to the dataset
        clear
        set obs 1
        gen age = `a'
        gen correlation = `corr'
        append using `corr_results'
        save `corr_results', replace
    }
    restore
}

preserve
* Load the results dataset and save to CSV
use `corr_results', clear
list
export delimited using "cont_MH_earn_corr.csv", replace
restore


preserve
* Create an empty dataset with the variables 'age' and 'correlation'
tempfile corr_results
clear
set obs 0
gen age = .
gen correlation = .
save `corr_results', replace
restore

* Loop over each age group and calculate correlation
levelsof age, local(ages)
foreach a of local ages {
    preserve
    keep if age == `a'
    
    * Check if there's data for the current age
    if _N > 0 {
        correlate MH labor_earnings if emp == 1
        matrix corr_matrix = r(C)
        local corr = corr_matrix[1,2]

        * Append this age's result to the dataset
        clear
        set obs 1
        gen age = `a'
        gen correlation = `corr'
        append using `corr_results'
        save `corr_results', replace
    }
    restore
}

preserve
* Load the results dataset and save to CSV
use `corr_results', clear
list
export delimited using "MH_earn_corr.csv", replace
restore