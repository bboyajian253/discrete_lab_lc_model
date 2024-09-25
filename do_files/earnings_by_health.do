// calculate the mean of earnings by health state and type. over all and then by age 

//by health state
preserve
collapse (mean) labor_income if emp == 1 & MH == 0
rename labor_income mean_earnings_MH0
tempfile mean_earnings_MH0
save `mean_earnings_MH0', replace
restore

preserve
collapse (mean) labor_income if emp == 1 & MH == 1
rename labor_income mean_earnings_MH1
tempfile mean_earnings_MH1
save `mean_earnings_MH1', replace
restore

//by health state and age
preserve
collapse (mean) labor_income if emp == 1 & MH == 0, by(age)
rename labor_income mean_earnings_MH0_age
tempfile mean_earnings_MH0_age
save `mean_earnings_MH0_age', replace
restore

preserve
collapse (mean) labor_income if emp == 1 & MH == 1, by(age)
rename labor_income mean_earnings_MH1_age
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
preserve
collapse (mean) labor_income if emp == 1 & MH_clust_k2 == 0
rename labor_income mean_earnings_MHT0
tempfile mean_earnings_MHT0
save `mean_earnings_MHT0', replace
restore

preserve
collapse (mean) labor_income if emp == 1 & MH_clust_k2 == 1
rename labor_income mean_earnings_MHT1
tempfile mean_earnings_MHT1
save `mean_earnings_MHT1', replace
restore

//by type and age
preserve
collapse (mean) labor_income if emp == 1 & MH_clust_k2 == 0, by(age)
rename labor_income mean_earnings_MHT0_age
tempfile mean_earnings_MHT0_age
save `mean_earnings_MHT0_age', replace
restore

preserve
collapse (mean) labor_income if emp == 1 & MH_clust_k2 == 1, by(age)
rename labor_income mean_earnings_MHT1_age
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

di ***End of earnings_by_health.do***