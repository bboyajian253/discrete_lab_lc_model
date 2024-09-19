cls

clear

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files" /*do file folder directory */
global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/my_model_2/input" /*output file folder directory */
global datadir  "$curdir/Data"

local dir = "$datadir"

cd "`dir'"
* Switch to data with MH and PH types calculated
use "`dir'/UKHLS_sample_probs", clear

drop if wage == .
drop if job_hours == .
drop if age == .

drop if wage < 10.0 // minimum wage in the uk is 11.44 euros for those over 21 years of age
drop if wage > 50.0 // set a high wage that we ignore people above, dont care about super high earners

xtile wage_percentiles = wage, nq(100)
xtile job_hours_percentile = job_hours, nq(100)

drop if job_hours_percentile <= 10 | job_hours_percentile >=90 // drop the bottom and top 10% of workers
//drop if wage_percentile >= 95

list wage if age == 74 
drop if wage > 30 & age == 74 // deals with a specfic outlier

* Calculate mean wages for MH = 0 and MH = 1
sum wage if MH == 0
local mean_wage0 = r(mean)
sum wage if MH == 1
local mean_wage1 = r(mean) + 1

* Calculate the difference
local difference = `mean_wage1' - `mean_wage0'

* Change to output directory
cd "$outdir"

* Preserve the original dataset
preserve
* Collapse the dataset to a single observation with the difference
gen diff = `difference'
collapse (mean) diff
export delimited using "MH_wage_moments.csv", replace
restore

* Function to collapse moments by age for a given mental health status
capture program drop collapse_moments
program define collapse_moments, rclass
    args mh_status

    * Verify MH variable exists
    capture confirm variable MH
    if _rc {
        display "MH variable not found"
		describe
        exit 1
    }

    preserve
    collapse (mean) emp if age >= 25 & age <= 75 & MH == `mh_status', by(age)
    tempfile emp_rate_data`mh_status'
    save `emp_rate_data`mh_status'', replace
    restore

    preserve
    collapse (sd) emp if age >= 25 & age <= 75 & MH == `mh_status', by(age)
    rename emp sd_emp
    tempfile emp_rate_sd_data`mh_status'
    save `emp_rate_sd_data`mh_status'', replace
    restore

    preserve
    collapse (mean) job_hours_decimal if age >= 25 & age <= 75 & MH == `mh_status', by(age)
    tempfile job_hours_data`mh_status'
    save `job_hours_data`mh_status'', replace
    restore

    preserve
    collapse (mean) wage if age >= 25 & age <= 75 & MH == `mh_status', by(age)
    tempfile wages_data`mh_status'
    save `wages_data`mh_status'', replace
    restore

    preserve
    collapse (sd) wage if age >= 25 & age <= 75 & MH == `mh_status', by(age)
    rename wage sd_wage
    tempfile wages_sd_data`mh_status'
    save `wages_sd_data`mh_status'', replace
    restore
	
	preserve
    * Export the job_hours dataset to a CSV file
    cd "$outdir"
    use `job_hours_data`mh_status'', clear
    export delimited using "labor_moments_`mh_status'.csv", replace

    * Export the wages dataset to a CSV file
    use `wages_data`mh_status'', clear
    export delimited using "wage_moments_`mh_status'.csv", replace

    * Read the wages dataset into memory
    import delimited "wage_moments_`mh_status'.csv", clear

    * Add the standard deviation to the dataset as a new observation
    merge 1:1 age using `wages_sd_data`mh_status''
    export delimited using "wage_moments_`mh_status'.csv", replace

    * Export the employment rate dataset to a CSV file
    use `emp_rate_data`mh_status'', clear
    export delimited using "emp_rate_moments_`mh_status'.csv", replace

    * Read the employment rate dataset into memory
    import delimited "emp_rate_moments_`mh_status'.csv", clear

    * Add the standard deviation to the dataset as a new observation
    merge 1:1 age using `emp_rate_sd_data`mh_status''
    export delimited using "emp_rate_moments_`mh_status'.csv", replace
	restore
	describe

end

* Function to combine moments into a single CSV file
capture program drop combine_moments
program define combine_moments

    cd "$outdir"

    * Combine employment rate moments
    import delimited "emp_rate_moments_0.csv", clear
    rename emp emp_mh0
    merge 1:1 age using "emp_rate_moments_1.csv", keep(match) nogen
    rename emp emp_mh1
    export delimited using "emp_rate_moments_by_MH.csv", replace

    * Combine wage moments
    import delimited "wage_moments_0.csv", clear
    rename wage wage_mh0
    merge 1:1 age using "wage_moments_1.csv", keep(match) nogen
    rename wage wage_mh1
    export delimited using "wage_moments_by_MH.csv", replace

    * Combine job hours moments
    import delimited "labor_moments_0.csv", clear
    rename job_hours_decimal job_hours_mh0
    merge 1:1 age using "labor_moments_1.csv", keep(match) nogen
    rename job_hours_decimal job_hours_mh1
    export delimited using "labor_moments_by_MH.csv", replace

end

collapse_moments 1
collapse_moments 0
combine_moments
