local num_lags = 10
// local corr_vars "wage log_wage"
local corr_vars "wage"

foreach corr_var in `corr_vars' {
    forvalues i = 1/`num_lags' {
        capture drop L`i'_`corr_var'
        gen L`i'_`corr_var' = L`i'.`corr_var'
        di "`corr_var' L`i'_`corr_var'"
        corr `corr_var' L`i'_`corr_var' if emp==1 & L`i'.emp==1
        sum L`i'_`corr_var' if emp==1
    }
}

di "Done!"