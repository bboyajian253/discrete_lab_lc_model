// calculate the mean of earnings by health state and type. over all and then by age 
//by health state
preserve
collapse (mean) labor_earnings if emp == 1 & MH == 0
rename labor_earnings mean_earnings_MH0
tempfile mean_earnings_MH0
save `mean_earnings_MH0', replace
restore

preserve
collapse (mean) labor_earnings if emp == 1 & MH == 1
rename labor_earnings mean_earnings_MH1
tempfile mean_earnings_MH1
save `mean_earnings_MH1', replace
restore

//by health state and age
preserve
collapse (mean) labor_earnings if emp == 1 & MH == 0, by(age)
rename labor_earnings mean_earnings_MH0_age
tempfile mean_earnings_MH0_age
save `mean_earnings_MH0_age', replace
restore

preserve
collapse (mean) labor_earnings if emp == 1 & MH == 1, by(age)
rename labor_earnings mean_earnings_MH1_age
tempfile mean_earnings_MH1_age
save `mean_earnings_MH1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve 

use `mean_earnings_MH0', clear
merge 1:1 _n using `mean_earnings_MH1'
export delimited using "mean_earnings_by_health.csv", replace

use `mean_earnings_MH0_age', clear
merge 1:1 _n using `mean_earnings_MH1_age'
export delimited using "mean_earnings_by_health_age.csv", replace

restore


// do teh same for log labor earnings
//by health state
preserve
collapse (mean) log_labor_earnings if emp == 1 & MH == 0
rename log_labor_earnings mean_earnings_MH0
tempfile mean_log_earnings_MH0
save `mean_log_earnings_MH0', replace
restore

preserve
collapse (mean) log_labor_earnings if emp == 1 & MH == 1
rename log_labor_earnings mean_earnings_MH1
tempfile mean_log_earnings_MH1
save `mean_log_earnings_MH1', replace
restore

//by health state and age
preserve
collapse (mean) log_labor_earnings if emp == 1 & MH == 0, by(age)
rename log_labor_earnings mean_earnings_MH0_age
tempfile mean_log_earnings_MH0_age
save `mean_log_earnings_MH0_age', replace
restore

preserve
collapse (mean) log_labor_earnings if emp == 1 & MH == 1, by(age)
rename log_labor_earnings mean_earnings_MH1_age
tempfile mean_log_earnings_MH1_age
save `mean_log_earnings_MH1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve 

use `mean_log_earnings_MH0', clear
merge 1:1 _n using `mean_log_earnings_MH1'
export delimited using "mean_log_earnings_by_health.csv", replace

use `mean_log_earnings_MH0_age', clear
merge 1:1 _n using `mean_log_earnings_MH1_age'
export delimited using "mean_log_earnings_by_health_age.csv", replace

restore

// do the same for log_hours
//by health state
preserve
collapse (mean) log_hours_decimal if emp == 1 & MH == 0
rename log_hours mean_log_hours_MH0
tempfile mean_log_hours_MH0
save `mean_log_hours_MH0', replace
restore

preserve
collapse (mean) log_hours_decimal if emp == 1 & MH == 1
rename log_hours mean_log_hours_MH1
tempfile mean_log_hours_MH1
save `mean_log_hours_MH1', replace
restore

//by health state and age
preserve
collapse (mean) log_hours_decimal if emp == 1 & MH == 0, by(age)
rename log_hours mean_log_hours_MH0_age
tempfile mean_log_hours_MH0_age
save `mean_log_hours_MH0_age', replace
restore

preserve
collapse (mean) log_hours_decimal if emp == 1 & MH == 1, by(age)
rename log_hours mean_log_hours_MH1_age
tempfile mean_log_hours_MH1_age
save `mean_log_hours_MH1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve

use `mean_log_hours_MH0', clear
merge 1:1 _n using `mean_log_hours_MH1'
export delimited using "mean_log_hours_by_health.csv", replace

use `mean_log_hours_MH0_age', clear
merge 1:1 _n using `mean_log_hours_MH1_age'
export delimited using "mean_log_hours_by_health_age.csv", replace

restore

// do the same for hours
//by health state
preserve
collapse (mean) job_hours_decimal if emp == 1 & MH == 0
rename job_hours_decimal mean_hours_MH0
tempfile mean_hours_MH0
save `mean_hours_MH0', replace
restore

preserve
collapse (mean) job_hours_decimal if emp == 1 & MH == 1
rename job_hours_decimal mean_hours_MH1
tempfile mean_hours_MH1
save `mean_hours_MH1', replace
restore

//by health state and age
preserve
collapse (mean) job_hours_decimal if emp == 1 & MH == 0, by(age)
rename job_hours_decimal mean_hours_MH0_age
tempfile mean_hours_MH0_age
save `mean_hours_MH0_age', replace
restore

preserve
collapse (mean) job_hours_decimal if emp == 1 & MH == 1, by(age)
rename job_hours_decimal mean_hours_MH1_age
tempfile mean_hours_MH1_age
save `mean_hours_MH1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve

use `mean_hours_MH0', clear
merge 1:1 _n using `mean_hours_MH1'
export delimited using "mean_hours_by_health.csv", replace

use `mean_hours_MH0_age', clear
merge 1:1 _n using `mean_hours_MH1_age'
export delimited using "mean_hours_by_health_age.csv", replace

restore

// do the same for log wages
// by health state
preserve
collapse (mean) log_wage if emp == 1 & MH == 0
rename log_wage mean_log_wage_MH0
tempfile mean_log_wage_MH0
save `mean_log_wage_MH0', replace
restore

preserve
collapse (mean) log_wage if emp == 1 & MH == 1
rename log_wage mean_log_wage_MH1
tempfile mean_log_wage_MH1
save `mean_log_wage_MH1', replace
restore

//by health state and age
preserve
collapse (mean) log_wage if emp == 1 & MH == 0, by(age)
rename log_wage mean_log_wage_MH0_age
tempfile mean_log_wage_MH0_age
save `mean_log_wage_MH0_age', replace
restore

preserve
collapse (mean) log_wage if emp == 1 & MH == 1, by(age)
rename log_wage mean_log_wage_MH1_age
tempfile mean_log_wage_MH1_age
save `mean_log_wage_MH1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve

use `mean_log_wage_MH0', clear
merge 1:1 _n using `mean_log_wage_MH1'
export delimited using "mean_log_wage_by_health.csv", replace

use `mean_log_wage_MH0_age', clear
merge 1:1 _n using `mean_log_wage_MH1_age'
export delimited using "mean_log_wage_by_health_age.csv", replace

restore

// do the same for wages
// by health state
preserve
collapse (mean) wage if emp == 1 & MH == 0
rename wage mean_wage_MH0
tempfile mean_wage_MH0
save `mean_wage_MH0', replace
restore

preserve
collapse (mean) wage if emp == 1 & MH == 1
rename wage mean_wage_MH1
tempfile mean_wage_MH1
save `mean_wage_MH1', replace
restore

//by health state and age
preserve
collapse (mean) wage if emp == 1 & MH == 0, by(age)
rename wage mean_wage_MH0_age
tempfile mean_wage_MH0_age
save `mean_wage_MH0_age', replace
restore

preserve
collapse (mean) wage if emp == 1 & MH == 1, by(age)
rename wage mean_wage_MH1_age
tempfile mean_wage_MH1_age
save `mean_wage_MH1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve

use `mean_wage_MH0', clear
merge 1:1 _n using `mean_wage_MH1'
export delimited using "mean_wage_by_health.csv", replace

use `mean_wage_MH0_age', clear
merge 1:1 _n using `mean_wage_MH1_age'
export delimited using "mean_wage_by_health_age.csv", replace

restore
// by type
local MH_clust MH_clust_50p_age
preserve
collapse (mean) labor_earnings if emp == 1 & `MH_clust' == 0
rename labor_earnings mean_earnings_MHT0
tempfile mean_earnings_MHT0
save `mean_earnings_MHT0', replace
restore

preserve
collapse (mean) labor_earnings if emp == 1 & `MH_clust' == 1
rename labor_earnings mean_earnings_MHT1
tempfile mean_earnings_MHT1
save `mean_earnings_MHT1', replace
restore

//by type and age
preserve
collapse (mean) labor_earnings if emp == 1 & `MH_clust' == 0, by(age)
rename labor_earnings mean_earnings_MHT0_age
tempfile mean_earnings_MHT0_age
save `mean_earnings_MHT0_age', replace
restore

preserve
collapse (mean) labor_earnings if emp == 1 & `MH_clust' == 1, by(age)
rename labor_earnings mean_earnings_MHT1_age
tempfile mean_earnings_MHT1_age
save `mean_earnings_MHT1_age', replace
restore

// * Merge and save
cd "$outdir"
preserve

use `mean_earnings_MHT0', clear
merge 1:1 _n using `mean_earnings_MHT1'
export delimited using "mean_earnings_by_health_type.csv", replace

use `mean_earnings_MHT0_age', clear
merge 1:1 _n using `mean_earnings_MHT1_age'
export delimited using "mean_earnings_by_health_type_age.csv", replace
restore



// By earnings level (decile, quintile, etc) what is the percentage in bad health state.

// gen earninngs deciles
gen labor_earnings_nonmissing = labor_earnings if labor_earnings != . & emp == 1
capture drop earnings_Q10
xtile earnings_Q10 = labor_earnings_nonmissing [pweight=wght], nq(10)
drop labor_earnings_nonmissing  

cd "$outdir"

preserve
drop if missing(earnings_Q10)
collapse (mean) badMH, by(earnings_Q10)
rename badMH mean_badMH
tempfile mean_bad_MH_by_earnings
save `mean_bad_MH_by_earnings', replace
export delimited using "mean_bad_MH_by_earnings.csv", replace
restore

di ***End of earnings_by_health.do***