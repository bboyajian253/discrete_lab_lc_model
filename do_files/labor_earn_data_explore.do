
local if_cond = "emp == 1 & log_labor_earnings != . & labor_earnings > 0"

preserve
collapse (min) log_labor_earnings if `if_cond', by(age)
rename log_labor_earnings min_log_labor_earnings
tempfile min_log_labor_earnings
save `min_log_labor_earnings', replace
list
restore

preserve
collapse (min) labor_earnings if `if_cond', by(age)
rename labor_earnings min_labor_earnings
tempfile min_labor_earnings
save `min_labor_earnings', replace
list 
restore

preserve
collapse (max) log_labor_earnings if `if_cond', by(age)
rename log_labor_earnings max_log_labor_earnings
tempfile max_log_labor_earnings
save `max_log_labor_earnings', replace
list 
restore

preserve
collapse (max) labor_earnings if `if_cond', by(age)
rename labor_earnings max_labor_earnings
tempfile max_labor_earnings
save `max_labor_earnings', replace
list 
restore

preserve
// load the first temp file
use `min_log_labor_earnings', clear
// merge with the rest of the temp files
merge 1:1 age using `min_labor_earnings'
drop _merge
merge 1:1 age using `max_log_labor_earnings'
drop _merge
merge 1:1 age using `max_labor_earnings'
drop _merge
list
restore


