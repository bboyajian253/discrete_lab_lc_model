cls

clear

// global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
// global dodir    "$curdir/Do_Files" /*do file folder directory */
// global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/model_uncert/input/MH_trans/" /*output file folder directory */
// global datadir  "$curdir/Data"
// local outdir = 

local dir = "$datadir"

cd "`dir'"
// use "`dir'/UKHLS_sample_trans", clear	
use "`dir'/UKHLS_sample_trans", clear

local start_age = 25
local end_age = 75

* Calculate 1-year UNCONDITIONAL binary health state transitions
local trans "P2G G2P"

* Initialize a Mata matrix with 0 columns
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
	
	mata: st_matrix("temp_matrix`MH_clus_ind'_`tran'", temp_matrix)

	preserve
	clear
	local outfile "$outdir/MH_trans_`tran'.csv"
	matrix list temp_matrix_`tran'
	svmat temp_matrix_`tran', names(col)
	if "`tran'" == "P2G"{
		local c1_name "MH_type_P2P"
		local c2_name "MH_type_P2G"
	}
	else {
		local c1_name "MH_type_G2P"
		local c2_name "MH_type_G2G"
	}
	rename c1 age
	rename c2 `c1_name'
	rename c3 `c2_name'
	*export delimited using "`outfile'", replace
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
merge 1:1 age using `tempfile_P2G'
drop _merge
merge 1:1 age using `tempfile_G2P'
drop _merge
export delimited using "`outfile'",replace
restore


****By cluster****
* Initialize a Mata matrix with 0 columns
mata: M = J(1, 0, .)

* Loop through each transition to calculate means
local MH_clust_types "MH_clust_50p MH_clust_k2"
local MH_clust_vals 0 1
foreach clust_type in `MH_clust_types'{
	foreach clust_val in `MH_clust_vals'{
		foreach tran in `trans' {

			sum MH_`tran' if `start_age' <= age & age <= `end_age' & `clust_type' == `clust_val'
			local MH_`tran'_`clust_type'_mean = r(mean)
			
			* Calculate the complement of the mean
			local one_min_`tran'_`clust_type'_mean = 1 - `MH_`tran'_`clust_type'_mean'

			* Store the results in Mata
			if "`tran'" == "P2G"{
				mata: M = M, `one_min_`tran'_`clust_type'_mean'
				mata: M = M, `MH_`tran'_`clust_type'_mean'
			}
			else{
				mata: M = M, `MH_`tran'_`clust_type'_mean'
				mata: M = M, `one_min_`tran'_`clust_type'_mean'
			}
		}
	}
	
    * Export the Mata matrix to a CSV file with column names
    mata: st_matrix("M", M)
	matrix colname M = "MH0_P2P" "MH0_P2G" "MH0_G2P" "MH0_G2G" "MH1_P2P" "MH1_P2G" "MH1_G2P" "MH1_G2G"
    * Export using the fixed file name pattern
	preserve
	clear
    svmat M, names(col)
    export delimited using "$outdir/MH_trans_by_`clust_type'.csv", replace
	di "**** Got here 5 ****"
    * Clear the Mata matrix for the next iteration
	restore
    mata: M = J(1, 0, .)
}

* By age and K Means MH index cluster
foreach clust_type in `MH_clust_types'{
	forvalues i = 1/2{
		foreach tran in `trans'{
			local MH_clus_ind = `i' - 1
			tab age MH_`tran' [aweight=wght0] if `start_age' <= age & age <= `end_age' & `clust_type' == `MH_clus_ind', matcell(xx) matrow(yy) matcol(zz)

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
	local outfile "$outdir/MH_trans_by_`clust_type'_age.csv"
	use `tempfile_0_P2G'
	merge 1:1 age using `tempfile_0_G2P'
	drop _merge
	merge 1:1 age using `tempfile_1_P2G'
	drop _merge
	merge 1:1 age using `tempfile_1_G2P'
	drop _merge
	export delimited using "`outfile'",replace
	restore

}


cd "`dir'"
save UKHLS_sample_trans, replace
