clear

// global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
// global dodir    "$curdir/Do_Files" /*do file folder directory */
// global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input/MH_trans/" /*output file folder directory */
// global datadir  "$curdir/Data"

local dir = "$datadir"

cd "`dir'"
use "`dir'/UKHLS_sample_trans", clear

local start_age = 25
local end_age = 75

// sort indiv_id year
// gen L1_mental_health = L1.mental_health
sort indiv_id year
xtset indiv_id year
capture drop L1_mental_health
gen L1_mental_health = L1.mental_health

local y_var "mental_health"
local age_cubic "age age2 age3"
local dem_controls "i.mar_stat i.sex i.educ i.race i.year" 
local weights_age_range "[pweight=wght] if `start_age' <= age & age <= `end_age'"

local control_spec1 " `age_cubic' `weights_age_range' "
local control_spec2 " `dem_controls' `control_spec1'"

local control_spec_list "control_spec1 control_spec2"  

local indep_vars_list `"L1_mental_health i.MH_clust_50p i.MH_clust_k2 "i.MH_clust_50p L1_mental_health" "i.MH_clust_k2 L1_mental_health""'

local num_controls : word count `control_spec_list'
local num_indep_vars : word count `indep_vars_list'

* Initialize the matrix with dimensions: rows = number of control specs, columns = number of independent vars
mata: M = J(`num_controls', `num_indep_vars' + 1, .)

local row = 1  // Initialize row index for the matrix

foreach control_spec in `control_spec_list' {
    local col = 1  // Initialize column index for the matrix
    foreach indep_var in `indep_vars_list' {
        reg `y_var' `indep_var' ``control_spec''
        local my_r2 = e(r2)
        mata: M[`row', `col'] = `my_r2'  // Store R-squared value in the current row and column
        local col = `col' + 1  // Move to the next column
    }
    local row = `row' + 1  // Move to the next row after completing the inner loop
}

mata: M
*** Controls only ***
reg `y_var' `control_spec2'
local my_r2 = e(r2)
mata: M[`row' - 1, `col'] = `my_r2'  // Store R-squared value for controls only

* Export the Mata matrix to a CSV file with column names
mata: st_matrix("M", M)
mata: M
preserve
clear
svmat M, names(col)
rename c1 lag_MH
rename c2 MH_type_50p
rename c3 MH_type_k2
rename c4 lag_MH_type_50p
rename c5 lag_MH_type_k2
rename c6 controls_only
export delimited using "$outdir/MH_type_reg_r2.csv", replace
restore