
// levelsof age, local(ages)
// preserve
// * Create an empty dataset with the variables 'age' and 'correlation'
// tempfile corr_results
// clear
// set obs 0
// gen lag = .
// gen correlation = .
// save `corr_results', replace
// restore

xtset indiv_id age	
local num_lags = 10
local corr_vars = "wage log_wage job_hours log_hours labor_earnings log_labor_earnings mental_health physical_health MH PH"
local num_vars : word count "`corr_vars'"

// Initialize an empty Mata matrix to store results, with num_lags + 1 rows and no columns initially
mata: ret_matrix1 = J(`num_lags' + 1, 0, .)
mata: ret_matrix2 = J(`num_lags' + 1, 0, .)

// Initialize a Stata local to store column names for CSV export
local col_names

foreach corr_var in `corr_vars' {
    mata: first_col = J(`num_lags' + 1, 1, .) // Initialize a column of missing values
    correlate `corr_var' 
    matrix corr_mat = r(C)
    // load mat
    mata: corr_mat = st_matrix("corr_mat")
    mata: coeff = corr_mat[1,1]
    mata: first_col[1] = coeff



    // Create lagged variables up to the number of lags
    forvalues i = 1/`num_lags' {
        capture drop L`i'_`corr_var'
        gen L`i'_`corr_var' = L`i'.`corr_var'

        local if_cond = ""
        if "`corr_var'" == "wage" | "`corr_var'" == "log_wage" | "`corr_var'" == "job_hours" | "`corr_var'" == "log_hours" ///
            | "`corr_var'" == "labor_earnings" | "`corr_var'" == "log_labor_earnings" {
            local if_cond = "if emp==1 & L`i'.emp==1"
        }
        di "if condition for `corr_var' is: `if_cond'"

        correlate `corr_var' L`i'_`corr_var' `if_cond'
        matrix corr_mat = r(C)
        // load mat
        mata: corr_mat = st_matrix("corr_mat")
        mata: coeff = corr_mat[2,1]
        mata: first_col[`i' + 1] = coeff
    }

    //print the first column
    mata: first_col
    // append to the second matrix
    mata: ret_matrix2 = ret_matrix2, first_col


    local if_cond = ""
    if "`corr_var'" == "wage" | "`corr_var'" == "log_wage" | "`corr_var'" == "job_hours" | "`corr_var'" == "log_hours" ///
        | "`corr_var'" == "labor_earnings" | "`corr_var'" == "log_labor_earnings" {
        local if_cond = "if emp==1 & L1.emp==1 & L2.emp==1 & L3.emp==1 & L4.emp==1 & L5.emp==1 & L6.emp==1 & L7.emp==1 & L8.emp==1 & L9.emp==1 & L10.emp==1"
    }
    di "if condition for `corr_var' is: `if_cond'"

    // Correlate the variable with its lagged versions
    correlate `corr_var' L1_`corr_var' L2_`corr_var' L3_`corr_var' L4_`corr_var' ///
        L5_`corr_var' L6_`corr_var' L7_`corr_var' L8_`corr_var' L9_`corr_var' L10_`corr_var' `if_cond'

    // Store the correlation matrix
    matrix corr_matrix = r(C)
    // Use Mata to extract the first column and append it to autocorr_matrix
    mata: corr_matrix = st_matrix("corr_matrix")       // Load Stata matrix into Mata
    mata: first_col = corr_matrix[,1]                  // Extract the first column

    // mata: autocorr_matrix = autocorr_matrix, first_col // Append first_col to autocorr_matrix
    mata: ret_matrix1 = ret_matrix1, first_col

    // Add the current variable name to the list of column names
    local col_names "`col_names' `corr_var'"
}

mata: ret_matrix1
mata: ret_matrix2

// Export the combined matrix to CSV
mata: st_matrix("ret_matrix1", ret_matrix1)  // Store Mata matrix back into Stata matrix
// Again for the second matrix
mata: st_matrix("ret_matrix2", ret_matrix2)  // Store Mata matrix back into Stata matrix

// // Save the autocorr_matrix to CSV with the appropriate column names
cd "$outdir"
preserve
clear
matrix colnames ret_matrix1 = `col_names'
// Use svmat to convert the Stata matrix into variables
svmat ret_matrix1, names(col)
export delimited using autocorr_matrix.csv, replace
restore


// Save the autocorr_matrix to CSV with the appropriate column names
cd "$outdir"
preserve
clear
matrix colnames ret_matrix2 = `col_names'
// Use svmat to convert the Stata matrix into variables
svmat ret_matrix2, names(col)
export delimited using autocorr_matrix_indiv_lag.csv, replace
restore

di "*** End of script: autocorr.do ***"
di "************ GOT TO THE END OF THE SCRIPT ************"