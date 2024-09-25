
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


// xtset indiv_id age	
// local num_lags = 10
// local corr_vars = "labor_income mental_health physical_health MH PH"
// // make a mata matrix with as many rows as the number of variables 
// local num_vars = word count "`corr_vars'"
// mata: autocorr_matrix = J(`num_vars', `num_lags' + 1 , .)

// foreach corr_var in `corr_vars'{

//     forvalues i = 1/`num_lags' {
        
//         capture drop L`i'_`corr_var' 
        
//         gen L`i'_`corr_var' = L`i'.`corr_var'
//     }

//     correlate `corr_var' L1_`corr_var' L2_`corr_var' L3_`corr_var' L4_`corr_var' ///
//     L5_`corr_var' L6_`corr_var' L7_`corr_var' L8_`corr_var' L9_`corr_var' L10_`corr_var'    

//     matrix corr_matrix = r(C)
//     // local corr = corr_matrix[1,]
//     // get the first column of the correlation matrix
//     // local corr = corr_matrix[1,2..]
//     * Use Mata to extract the first column

//     * Store the correlation matrix
//     matrix corr_matrix = r(C)

//     * Use Mata to extract the first column (Stata 14 compatible)
//     mata: corr_matrix = st_matrix("corr_matrix")  // Load Stata matrix into Mata
//     mata: first_col = corr_matrix[,1]             // Extract the first column
//     mata: st_matrix("autocorr_`corr_var'", first_col)       // Store the first column back into a Stata matrix

//     * Now you can access the first column in Stata as a matrix or loop over the values
//     matrix list first_col

// }

xtset indiv_id age	
local num_lags = 10
local corr_vars = "labor_income mental_health physical_health MH PH"
local num_vars : word count "`corr_vars'"

// Initialize an empty Mata matrix to store results, with num_lags + 1 rows and no columns initially
mata: autocorr_matrix = J(`num_lags' + 1, 0, .)

// Initialize a Stata local to store column names for CSV export
local col_names

foreach corr_var in `corr_vars' {

    // Create lagged variables up to the number of lags
    forvalues i = 1/`num_lags' {
        capture drop L`i'_`corr_var'
        gen L`i'_`corr_var' = L`i'.`corr_var'
    }

    // Correlate the variable with its lagged versions
    correlate `corr_var' L1_`corr_var' L2_`corr_var' L3_`corr_var' L4_`corr_var' ///
        L5_`corr_var' L6_`corr_var' L7_`corr_var' L8_`corr_var' L9_`corr_var' L10_`corr_var'

    // Store the correlation matrix
    matrix corr_matrix = r(C)

    // Use Mata to extract the first column and append it to autocorr_matrix
    mata: corr_matrix = st_matrix("corr_matrix")       // Load Stata matrix into Mata
    mata: first_col = corr_matrix[,1]                  // Extract the first column
    mata: autocorr_matrix = autocorr_matrix, first_col // Append first_col to autocorr_matrix

    // Add the current variable name to the list of column names
    local col_names "`col_names' `corr_var'"
}

// Export the combined matrix to CSV
mata: st_matrix("autocorr_matrix", autocorr_matrix)  // Store Mata matrix back into Stata matrix

// Save the autocorr_matrix to CSV with the appropriate column names
cd "$outdir"
matrix colnames autocorr_matrix = `col_names'
mat2txt, matrix(autocorr_matrix) saving(autocorr_matrix.csv) replace

// outsheet using autocorr_matrix.csv, names replace

