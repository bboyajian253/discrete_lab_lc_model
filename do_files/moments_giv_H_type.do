cls

clear

local dir = "$datadir"

cd "`dir'"
// use "`dir'/UKHLS_sample_trans", clear
use "`dir'/UKHLS_sample_trans", clear

local start_age = 25
local end_age = 75

* Collapse moments by age for ages `start_age'-`end_age' and save the results
* employmemnt emp
preserve
collapse (mean) emp if age >= `start_age' & age <= `end_age', by(age)
tempfile emp_rate_data
save `emp_rate_data', replace
restore

preserve
collapse (sd) emp if age >= `start_age' & age <= `end_age', by(age)
rename emp sd_emp
tempfile emp_rate_sd_data
save `emp_rate_sd_data', replace
restore

* Collapse moments by age for ages `start_age'-`end_age' and save the results
preserve
collapse (mean) job_hours_decimal if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
tempfile job_hours_data
save `job_hours_data', replace
restore

* Collapse job_hours_decimal to sd by age for ages `start_age'-`end_age' and save the results
preserve
collapse (sd) job_hours_decimal if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
rename job_hours_decimal sd_job_hours_decimal
tempfile job_hours_sd_data
save `job_hours_sd_data', replace
restore

// Same for log hours
preserve
collapse (mean) log_hours if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
tempfile log_hours_data
save `log_hours_data', replace
restore

preserve
collapse (sd) log_hours if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
rename log_hours sd_log_hours
tempfile log_hours_sd_data
save `log_hours_sd_data', replace
restore



* Change to output directory
cd "$outdir"


// may want to run a regression here to isolate the effect of MH on wages, accounting for other factors represented in the model e.g. an age quadratic/cubic
// capture drop age2
// gen age2 = age*age
// local spec = "i.MH i.MH_clust_50p_age age age2 i.year i.educ i.mar_stat i.sex"
// local spec = "i.MH i.MH_clust_50p_age i.MH#i.MH_clust_50p_age age age2 i.year i.educ"
// reg emp `spec'
// reg log_labor_earnings `spec' if emp == 1
// reg log_hours `spec' if emp == 1
// reg log_wage `spec' if emp == 1


sum job_hours_decimal if emp == 1
local mean_hours = r(mean)

preserve
gen mean_hours = `mean_hours'
collapse (mean) mean_hours
tempfile mean_hours_data
save `mean_hours_data', replace
restore

preserve
use `mean_hours_data', clear
export delimited using "alpha_mom_targ.csv", replace
restore

* Calculate mean wages for MH = 0 and MH = 1
sum log_wage if MH == 0 & emp == 1 
local mean_wage0 = r(mean) 
sum log_wage if MH == 1 & emp == 1 
local mean_wage1 = r(mean)

* Calculate the difference
// local difference = log(`mean_wage1') - log(`mean_wage0')
local difference = `mean_wage1' - `mean_wage0'
di `difference'

* Preserve the original dataset
preserve
* Collapse the dataset to a single observation with the difference
gen diff = `difference'
collapse (mean) diff
rename diff MH_diff_lnwage
tempfile wage_diff_data
save `wage_diff_data', replace
// export delimited using "MH_wage_moments.csv", replace
restore

// local spec = "i.MH i.MH_clust_50p_age i.MH#i.MH_clust_50p_age age age2 i.year i.educ"
local spec = "i.MH age age2 i.year" 
reg log_wage `spec' if emp == 1
// Store the coefficient for MH == 1
local mycoeff = _b[1.MH]
di "Coefficient for MH == 1: `mycoeff'"

* Preserve the original dataset and collapse to a single observation with the coefficient
preserve
gen MH_coeff_lnwage = `mycoeff'
collapse (mean) MH_coeff_lnwage
tempfile coeff_data
save `coeff_data', replace
restore

// merge and store
preserve
* Cross both datasets to combine them
use `wage_diff_data', clear
cross using `coeff_data'
* Export the combined dataset to CSV
export delimited using "MH_wage_moments.csv", replace
restore

// local retirement_age = 60
local pre_retirement_age = 55

// calculate mean hours for MH = 0 and MH = 1
sum log_hours_decimal if MH == 0 & emp == 1 & age < `pre_retirement_age'
local mean_log_hours0 = r(mean)
sum log_hours_decimal if MH == 1 & emp == 1 & age < `pre_retirement_age'
local mean_log_hours1 = r(mean)

// calculate the difference
local difference_hours = `mean_log_hours1' - `mean_log_hours0'

// preserve the original dataset
preserve
// collapse the dataset to a single observation with the difference
gen MH_diff_hours = `difference_hours'
collapse (mean) MH_diff_hours
tempfile hours_diff_data
save `hours_diff_data', replace
// export delimited using "MH_hours_moments.csv", replace
restore

reg log_hours_decimal `spec' if emp == 1 & age < `pre_retirement_age'
// store the coefficient for MH == 1
local mycoeff_hours = _b[1.MH]
di "Coefficient for MH == 1: `mycoeff_hours'"

// preserve the original dataset and collapse to a single observation with the coefficient
preserve
gen MH_coeff_lnhours = `mycoeff_hours'
collapse (mean) MH_coeff_lnhours
tempfile coeff_hours_data
save `coeff_hours_data', replace
restore

// merge and store
preserve
// cross both datasets to combine them
use `hours_diff_data', clear
cross using `coeff_hours_data'
// export the combined dataset to CSV
export delimited using "MH_hours_moments.csv", replace
restore

preserve
collapse (mean) wage if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
list
tempfile wages_data
save `wages_data', replace
restore

* Collapse wage to sd by age for ages `start_age'-`end_age' and save the results
preserve
collapse (sd) wage if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
rename wage sd_wage
tempfile wages_sd_data
save `wages_sd_data', replace
restore

// same for log wages
preserve
collapse (mean) log_wage if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
tempfile log_wages_data
save `log_wages_data', replace
restore

preserve
collapse (sd) log_wage if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
rename log_wage sd_log_wage
tempfile log_wages_sd_data
save `log_wages_sd_data', replace
restore

* Mean and sd of labor earnings by age
preserve
collapse (mean) labor_earnings if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
tempfile earnings_data
save `earnings_data', replace
restore

preserve
collapse (sd) labor_earnings if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
rename labor_earnings sd_labor_earnings
tempfile earnings_sd_data
save `earnings_sd_data', replace
restore

// same for log labor earnings
preserve
collapse (mean) log_labor_earnings if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
tempfile log_earnings_data
save `log_earnings_data', replace
restore

preserve
collapse (sd) log_labor_earnings if emp == 1 & age >= `start_age' & age <= `end_age', by(age)
rename log_labor_earnings sd_log_labor_earnings
tempfile log_earnings_sd_data
save `log_earnings_sd_data', replace
restore

// merge and store earnings moments
preserve
use `earnings_data', clear
merge 1:1 age using `earnings_sd_data', nogen
merge 1:1 age using `log_earnings_data', nogen
merge 1:1 age using `log_earnings_sd_data', nogen
export delimited using "earnings_moments.csv", replace
restore

// merge and store wage and log wage moments
preserve
use `job_hours_data', clear
merge 1:1 age using `job_hours_sd_data', nogen
merge 1:1 age using `log_hours_data', nogen
merge 1:1 age using `log_hours_sd_data', nogen
export delimited using "labor_moments.csv", replace
restore

preserve
* Step 4: Export the wages dataset to a CSV file
use `wages_data', clear
merge 1:1 age using `wages_sd_data', nogen
merge 1:1 age using `log_wages_data', nogen
merge 1:1 age using `log_wages_sd_data', nogen
export delimited using "wage_moments.csv", replace
restore

preserve
* Export the wages dataset to a CSV file
use `emp_rate_data', clear
export delimited using "emp_rate_moments.csv", replace

* Read the wages dataset into memory
import delimited "emp_rate_moments.csv", clear

* Add the standard deviation to the dataset as a new observation
* gen sd_wage = `initial_sd_wage'
merge 1:1 age using `emp_rate_sd_data', nogen
 
* Export the dataset to a CSV file
export delimited using "emp_rate_moments.csv", replace
restore

***************************
***Reload untrimmed data***
***************************

use "`dir'/UKHLS_sample_trans", clear

* For a given clustering method
// local MH_clust MH_clust_k2
local MH_clust MH_clust_50p_age

mata: M = J(1, 0, .)

sum `MH_clust' if age == `start_age' 
local share_in_high_type = r(mean)

mata: M = M, 1 - `share_in_high_type'
mata: M = M, `share_in_high_type' 

local types 0 1

foreach MH_type in `types'{
	sum MH if `MH_clust' == `MH_type' & age == `start_age'
	local share_in_good_state = r(mean)
	mata: M = M, 1 - `share_in_good_state'
	mata: M = M, `share_in_good_state'
	
}

mata: M

* Export the Mata matrix to a CSV file with column names
mata: st_matrix("M", M)
matrix colname M = "Share_low_type" "Share_high_type" "Share_bad_type0" "Share_good_type0" "Share_bad_type1" "Share_good_type1"
preserve
clear
svmat M, names(col)
export delimited using "$outdir/`MH_clust'_pop_shares.csv", replace
restore

di `MH_clust'

sum mental_health if `MH_clust' == 0
local mean_mental_health_bad = r(mean)
local n_MH_clust0 = r(N)

sum mental_health if `MH_clust' == 1
local mean_mental_health_good = r(mean)
local n_MH_clust1 = r(N)

local mean_mental_health_type_diff = `mean_mental_health_good' - `mean_mental_health_bad'

gen u_H = 0
replace u_H = `mean_mental_health_type_diff' if `MH_clust' == 1
sum u_H

gen adj_MH_ind = mental_health - u_H

xtset indiv_id age
gen L1_adj_MH_ind = L.adj_MH_ind
reg adj_MH_ind L1_adj_MH_ind
reg adj_MH_ind L1_adj_MH_ind if `MH_clust' == 0
reg adj_MH_ind L1_adj_MH_ind if `MH_clust' == 1
*xtreg adj_MH_ind L1_adj_MH_ind, re
*xtreg adj_MH_ind L1_adj_MH_ind, fe

* 1-years MH transition	
* gen MH_B2G = cond(MH==0 & F1.MH==1,1, cond(MH==0 & F1.MH==0,0,.))	
* gen MH_G2B = cond(MH==1 & F1.MH==0,1, cond(MH==1 & F1.MH==1,0,.))

* Calculate 1-year binary health state transitions
local trans "P2G G2P"

* Initialize a Mata matrix with 4 columns
mata: M = J(1, 0, .)
* Loop through each transition to calculate means
foreach tran in `trans' {
    sum MH_`tran' if `start_age' <= age & age <= `end_age' 
    local MH_`tran'_mean = r(mean)
    
    * Calculate the complement of the mean
    local one_min_`tran'_mean = 1 - `MH_`tran'_mean'

    * Store the results in Mata
	if "`tran'" == "P2G"{
		mata: M = M, `one_min_`tran'_mean'
		mata: M = M, `MH_`tran'_mean'
	}
	else{
		mata: M = M, `MH_`tran'_mean'
		mata: M = M, `one_min_`tran'_mean'
	}
}

* Export the Mata matrix to a CSV file with column names
mata: st_matrix("M", M)
matrix colname M = "MH_P2P" "MH_P2G" "MH_G2P" "MH_G2G"
preserve
clear
svmat M, names(col)
export delimited using "$outdir/MH_trans_uncond.csv", replace
restore

// By age
foreach tran in `trans'{
	tab age MH_`tran' [aweight=wght0] if `start_age' <= age & age <= `end_age', matcell(xx) matrow(yy) matcol(zz)

	* Convert matrices to Mata
	mata : xx = st_matrix("xx")
	mata : yy = st_matrix("yy")
	mata : xx = xx :/ rowsum(xx)
	mata : st_matrix("xx", xx)

	// Load matrices
	mata: xx = st_matrix("xx")
	mata: yy = st_matrix("yy")

	// Check dimensions and contents
	mata: num_rows = rows(xx)
	mata: num_cols = cols(xx)
	mata: st_numscalar("num_rows", num_rows)
	
	// Create a temporary matrix to store the output
	mata: temp_matrix = J(num_rows, num_cols + 1, .) // Adding one for age
	mata: temp_matrix[., 2..(num_cols + 1)] = xx
	mata: temp_matrix[., 1] = yy
	mata: temp_matrix
	
	if "`tran'" == "G2P" {
		// Swap the second and third columns
		mata: temp_matrix[., (2, 3)] = temp_matrix[., (3, 2)]
	}
	
	mata: st_matrix("temp_matrix_`tran'", temp_matrix)

	preserve
	clear
	// local outfile "$outdir/MH_trans_`tran'.csv"
	matrix list temp_matrix_`tran'
	svmat temp_matrix_`tran', names(col)
	if "`tran'" == "P2G"{
		local c1_name "MH_P2P"
		local c2_name "MH_P2G"
	}
	else {
		local c1_name "MH_G2P"
		local c2_name "MH_G2G"
	}
	rename c1 age
	rename c2 `c1_name'
	rename c3 `c2_name'
	// *export delimited using "`outfile'", replace
	tempfile tempfile_`tran'
	save `tempfile_`tran''
	restore
}

* Merge and store
preserve
clear 
local outfile "$outdir/MH_trans_uncond_age.csv"
use `tempfile_P2G'
merge 1:1 age using `tempfile_G2P'
drop _merge
export delimited using "`outfile'",replace
restore

*By K Means index cluster
* Initialize a Mata matrix with 0 columns
mata: M = J(1, 0, .)
* Loop through each transition to calculate means
local MH_clusters 0 1
foreach clust in `MH_clusters'{
	foreach tran in `trans' {
		sum MH_`tran' if `start_age' <= age & age <= `end_age' & `MH_clust' == `clust'
		local MH_`tran'_mean = r(mean)
		
		* Calculate the complement of the mean
		local one_min_`tran'_mean = 1 - `MH_`tran'_mean'

		* Store the results in Mata
		if "`tran'" == "P2G"{
			mata: M = M, `one_min_`tran'_mean'
			mata: M = M, `MH_`tran'_mean'
		}
		else{
			mata: M = M, `MH_`tran'_mean'
			mata: M = M, `one_min_`tran'_mean'
		}
	}
}

* Export the Mata matrix to a CSV file with column names
mata: st_matrix("M", M)
matrix colname M = "MH0_P2P" "MH0_P2G" "MH0_G2P" "MH0_G2G" "MH1_P2P" "MH1_P2G" "MH1_G2P" "MH1_G2G"
preserve
clear
svmat M, names(col)
export delimited using "$outdir/MH_trans_by_MH_clust.csv", replace
restore

* By age and K Means MH index cluster
forvalues i = 1/2{
	foreach tran in `trans'{
		local MH_clus_ind = `i' - 1
		tab age MH_`tran' [aweight=wght0] if `start_age' <= age & age <= `end_age' & `MH_clust' == `MH_clus_ind', matcell(xx) matrow(yy) matcol(zz)

		* Convert matrices to Mata
		mata : xx = st_matrix("xx")
		mata : yy = st_matrix("yy")
		mata : xx = xx :/ rowsum(xx)
		mata : st_matrix("xx", xx)

		// Load matrices
		mata: xx = st_matrix("xx")
		mata: yy = st_matrix("yy")

		// Check dimensions and contents
		mata: num_rows = rows(xx)
		mata: num_cols = cols(xx)
		mata: st_numscalar("num_rows", num_rows)
		
		// Create a temporary matrix to store the output
		mata: temp_matrix = J(num_rows, num_cols + 1, .) // Adding one for age
		mata: temp_matrix[., 2..(num_cols + 1)] = xx
		mata: temp_matrix[., 1] = yy
		mata: temp_matrix
		
		if "`tran'" == "G2P" {
			// Swap the second and third columns
			mata: temp_matrix[., (2, 3)] = temp_matrix[., (3, 2)]
		}
		
		mata: st_matrix("temp_matrix`MH_clus_ind'_`tran'", temp_matrix)

		preserve
		clear
		local outfile "$outdir/MH_trans`MH_clus_ind'_`tran'.csv"
		matrix list temp_matrix`MH_clus_ind'_`tran'
		svmat temp_matrix`MH_clus_ind'_`tran', names(col)
		if "`tran'" == "P2G"{
			local c1_name "MH_type_`MH_clus_ind'_P2P"
			local c2_name "MH_type_`MH_clus_ind'_P2G"
		}
		else {
			local c1_name "MH_type_`MH_clus_ind'_G2P"
			local c2_name "MH_type_`MH_clus_ind'_G2G"
		}
		rename c1 age
		rename c2 `c1_name'
		rename c3 `c2_name'
		*export delimited using "`outfile'", replace
		tempfile tempfile_`MH_clus_ind'_`tran'
		save `tempfile_`MH_clus_ind'_`tran''
		restore
	}
}
* Merge and store
preserve
clear 
local outfile "$outdir/MH_trans_by_MH_clust_age.csv"
use `tempfile_0_P2G'
merge 1:1 age using `tempfile_0_G2P'
drop _merge
merge 1:1 age using `tempfile_1_P2G'
drop _merge
merge 1:1 age using `tempfile_1_G2P'
drop _merge
export delimited using "`outfile'",replace
restore

// share of people in bad mental health state by age
preserve
collapse (mean) badMH, by(age)
rename badMH mean_badMH
tempfile mean_MH
save `mean_MH', replace
export delimited using "mean_bad_MH_by_age.csv", replace
restore

// share of people in bad physical health state by age
preserve
collapse (mean) badPH, by(age)
rename badPH mean_badPH
tempfile mean_PH
save `mean_PH', replace
export delimited using "mean_bad_PH_by_age.csv", replace
restore


cd "`dir'"
save UKHLS_sample_k2, replace
