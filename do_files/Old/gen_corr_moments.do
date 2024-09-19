* Create a new dataset to store correlations
tempfile corr_results
save `corr_results', replace

* Loop over each age group and calculate correlation
levelsof age, local(ages)
foreach a of local ages {
    preserve
    keep if age == `a'
    summarize
    local mean_labor_income = r(mean)
    local sd_labor_income = r(sd)
    local mean_log_labor_income = r(mean) if !missing(log_labor_income)
    local sd_log_labor_income = r(sd) if !missing(log_labor_income)
    restore
    * Calculate and save correlation
    tempfile temp_data
    save `temp_data', replace
    use `temp_data', clear
    correlate mental_health labor_income
    matrix list r(C)
    * Save results
    matrix corr_matrix = r(C)
    svmat corr_matrix, names(col)
    save `corr_results', replace
}

* Load the results dataset
use `corr_results', clear
list
