
cls
clear
local dir = "$datadir"

cd "`dir'"
// use "`dir'/UKHLS_sample_trans", clear
use "`dir'/UKHLS_sample_trans", clear

local start_age = 25
local end_age = 55
local age_range = `end_age' - `start_age' + 1

local drop_cond = "age < `start_age' | age > `end_age' | emp == 0"

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
collapse (mean) labor_earnings, by(age)
collapse (mean) labor_earnings
rename labor_earnings mean_lt_earnings
tempfile mean_lt_earnings
save `mean_lt_earnings', replace
restore

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
collapse (sd) labor_earnings, by(age)
collapse (mean) labor_earnings
rename labor_earnings sd_lt_earnings
tempfile sd_lt_earnings
save `sd_lt_earnings', replace
restore

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
egen p10 = pctile(labor_earnings), p(10) by(age)
sum labor_earnings if p10 == 1
collapse (mean) p10
rename p10 p10_lt_earnings
tempfile p10_lt_earnings
save `p10_lt_earnings', replace
restore

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
egen p25 = pctile(labor_earnings), p(25) by(age)
collapse (mean) p25
rename p25 p25_lt_earnings
tempfile p25_lt_earnings
save `p25_lt_earnings', replace
restore

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
egen p50 = pctile(labor_earnings), p(50) by(age)
collapse (mean) p50
rename p50 p50_lt_earnings
tempfile p50_lt_earnings
save `p50_lt_earnings', replace
restore

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
egen p75 = pctile(labor_earnings), p(75) by(age)
collapse (mean) p75
rename p75 p75_lt_earnings
tempfile p75_lt_earnings
save `p75_lt_earnings', replace
restore

preserve
drop if `drop_cond'
replace labor_earnings = labor_earnings * 12
egen p90 = pctile(labor_earnings), p(90) by(age)
collapse (mean) p90
rename p90 p90_lt_earnings
tempfile p90_lt_earnings
save `p90_lt_earnings', replace
restore

cd "$outdir"

preserve
use `mean_lt_earnings', clear
cross using `sd_lt_earnings'
cross using `p10_lt_earnings'
cross using `p25_lt_earnings'
cross using `p50_lt_earnings'
cross using `p75_lt_earnings'
cross using `p90_lt_earnings'
export delimited using "lt_earn_stats.csv", replace
restore