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