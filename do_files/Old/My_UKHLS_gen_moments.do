cls

clear

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files" /*do file folder directory */
global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input" /*output file folder directory */
global datadir  "$curdir/Data"

local dir = "$datadir"

cd "`dir'"
use "`dir'/UKHLS_sample_probs", clear

* Collapse moments by age for ages 25-75 and save the results
* employmemnt emp
preserve
collapse (mean) emp if age >= 25 & age <= 75, by(age)
tempfile emp_rate_data
save `emp_rate_data', replace
restore

preserve
collapse (sd) emp if age >= 25 & age <= 75, by(age)
rename emp sd_emp
tempfile emp_rate_sd_data
save `emp_rate_sd_data', replace
restore

sort age
quietly sum wage if age == 25
local initial_sd_wage = r(sd)

* Display the value of initial_sd_wage
display "Initial SD of wage for age 25: `initial_sd_wage'"

* Collapse moments by age for ages 25-75 and save the results
preserve
collapse (mean) job_hours_decimal if age >= 25 & age <= 75, by(age)
tempfile job_hours_data
save `job_hours_data', replace
restore

* Collapse wage and log_wage by age for ages 25-75 and save the results
preserve
collapse (mean) wage if age >= 25 & age <= 75, by(age)
tempfile wages_data
save `wages_data', replace
restore

* Collapse wage to sd by age for ages 25-75 and save the results
preserve
collapse (sd) wage if age >= 25 & age <= 75, by(age)
rename wage sd_wage
tempfile wages_sd_data
save `wages_sd_data', replace
restore

* Change to output directory
cd "$outdir"

* Export the job_hours dataset to a CSV file
use `job_hours_data', clear
export delimited using "labor_moments.csv", replace

* Step 4: Export the wages dataset to a CSV file
use `wages_data', clear
export delimited using "wage_moments.csv", replace

* Read the wages dataset into memory
import delimited "wage_moments.csv", clear

* Add the standard deviation to the dataset as a new observation
* gen sd_wage = `initial_sd_wage'
merge 1:1 age using `wages_sd_data'
 
* Export the dataset to a CSV file
export delimited using "wage_moments.csv", replace

* Export the wages dataset to a CSV file
use `emp_rate_data', clear
export delimited using "emp_rate_moments.csv", replace

* Read the wages dataset into memory
import delimited "emp_rate_moments.csv", clear

* Add the standard deviation to the dataset as a new observation
* gen sd_wage = `initial_sd_wage'
merge 1:1 age using `emp_rate_sd_data'
 
* Export the dataset to a CSV file
export delimited using "emp_rate_moments.csv", replace

* Switch to data with MH and PH types calculated
use "`dir'/UKHLS_sample_probs", clear

* Calculate mean wages for MH = 0 and MH = 1
sum wage if MH == 0
local mean_wage0 = r(mean)
sum wage if MH == 1
local mean_wage1 = r(mean)

* Calculate the difference
local difference = log(`mean_wage1') - log(`mean_wage0')

* Preserve the original dataset
preserve
* Collapse the dataset to a single observation with the difference
gen diff = `difference'
collapse (mean) diff
export delimited using "MH_wage_moments.csv", replace
restore

* K Means cluster
preserve
xtset indiv_id year
sort mental_health
collapse (mean) mental_health, by(indiv_id)
cluster kmeans mental_health, k(2)
rename _clus_1 MH_cluster
tempfile clusterdata
save `clusterdata'
restore

merge m:1 indiv_id using `clusterdata'
tab _merge
*drop _merge

sum mental_health if MH_cluster == 1
local mean_MH_1 = r(mean)
di `mean_MH_1'
sum mental_health if MH_cluster == 2
local mean_MH_2 = r(mean)
di `mean_MH_2'

if `mean_MH_1' >= `mean_MH_2'{
	replace MH_cluster = 0 if MH_cluster == 2
} 
else {
	replace MH_cluster = 0 if MH_cluster == 1
	replace MH_cluster = 1 if MH_cluster == 2
}

sum mental_health if MH_cluster == 0
local mean_mental_health_bad = r(mean)
local n_MH_clust0 = r(N)

sum mental_health if MH_cluster == 1
local mean_mental_health_good = r(mean)
local n_MH_clust1 = r(N)

local mean_mental_health_type_diff = `mean_mental_health_good' - `mean_mental_health_bad'
di `mean_mental_health_type_diff'
local share_bad_MH_clust = `n_MH_clust0'/(`n_MH_clust0' + `n_MH_clust1')
di `share_bad_MH_clust'

* Preserve the original dataset
preserve
* Collapse the dataset to a single observation with the difference
gen share_bad = `share_bad_MH_clust'
collapse (mean) share_bad
export delimited using "share_bad_MH_clust.csv", replace
restore


gen u_H = 0
replace u_H = `mean_mental_health_type_diff' if MH_cluster == 1
sum u_H

gen adj_MH_ind = mental_health - u_H

xtset indiv_id year
gen L1_adj_MH_ind = L.adj_MH_ind
reg adj_MH_ind L1_adj_MH_ind
reg adj_MH_ind L1_adj_MH_ind if MH_cluster == 0
reg adj_MH_ind L1_adj_MH_ind if MH_cluster == 1
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
    sum MH_`tran' if 25 <= age & age <= 75 
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

*By K Means index cluster
* Initialize a Mata matrix with 0 columns
mata: M = J(1, 0, .)
* Loop through each transition to calculate means
local MH_clusters 0 1
foreach clust in `MH_clusters'{
	foreach tran in `trans' {
		sum MH_`tran' if 25 <= age & age <= 75 & MH_cluster == `clust'
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
		tab age MH_`tran' [aweight=wght0] if 25 <= age & age <= 75 & MH_cluster == `MH_clus_ind', matcell(xx) matrow(yy) matcol(zz)

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

*1-year binary health state transitions
tab age MH_P2G [aweight=wght0] if 25 <= age & age<=75 & MH_cluster == 0, matcell(xx) matrow(yy) matcol(zz)
mata : xx = st_matrix("xx")
mata : xx = xx :/rowsum(xx)
mata : st_matrix("xx",xx)	
putexcel set "$outdir/MH_trans", sheet("Figure1_MH") modify	
putexcel  A1 = "Bad MH Type : % transition from bad mental health"
putexcel  A2 = "age"
putexcel  A3 = matrix(yy)  B2 = "to poor"  C2 = "to good"  
putexcel  B3 = matrix(xx)

*1-year binary health state transitions
tab age MH_P2G [aweight=wght0] if 25 <= age & age<=75 & MH_cluster == 1, matcell(xx) matrow(yy) matcol(zz)
mata : xx = st_matrix("xx")
mata : xx = xx :/rowsum(xx)
mata : st_matrix("xx",xx)	
putexcel set "$outdir/MH_trans", sheet("Figure1_MH") modify	
putexcel  H1 = "Good MH Type : % transition from bad mental health"
putexcel  H2 = "age"
putexcel  H3 = matrix(yy)  I2 = "to poor"  J2 = "to good"  
putexcel  I3 = matrix(xx)

*1-year binary health state transitions
tab age MH_G2P [aweight=wght0] if 25 <= age & age<=75 & MH_cluster == 0, matcell(xx) matrow(yy) matcol(zz)
mata : xx = st_matrix("xx")
mata : xx = xx :/rowsum(xx)
mata : st_matrix("xx",xx)	
putexcel set "$outdir/MH_trans", sheet("Figure1_MH") modify	
putexcel  N1 = "Bad MH Type : % transition from good mental health"
putexcel  N2 = "age"
putexcel  N3 = matrix(yy)  O2 = "to good"  P2 = "to poor"  
putexcel  O3 = matrix(xx)

*1-year binary health state transitions
tab age MH_G2P [aweight=wght0] if 25 <= age & age<=75 & MH_cluster == 1, matcell(xx) matrow(yy) matcol(zz)
mata : xx = st_matrix("xx")
mata : xx = xx :/rowsum(xx)
mata : st_matrix("xx",xx)	
putexcel set "$outdir/MH_trans", sheet("Figure1_MH") modify	
putexcel  T1 = "Good MH Type : % transition from good mental health"
putexcel  T2 = "age"
putexcel  T3 = matrix(yy)  U2 = "to good"  V2 = "to poor"  
putexcel  U3 = matrix(xx)






