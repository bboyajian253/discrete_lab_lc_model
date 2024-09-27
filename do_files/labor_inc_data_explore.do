
local if_cond = "emp == 1 & log_labor_earnings != . & labor_earnings > 1"

preserve
// collapse (min) labor_earnings, by(age)
// collapse (min) log_labor_earnings if emp == 1 & log_labor_earnings != . & labor_earnings > 0, by(age)
collapse (min) log_labor_earnings if `if_cond', by(age)
list log_labor_earnings
restore

preserve
// collapse (min) labor_earnings, by(age)
// collapse (min) labor_earnings if emp == 1 & log_labor_earnings != . & labor_earnings > 0, by(age)
collapse (min) labor_earnings if `if_cond', by(age)
list labor_earnings
restore

preserve
// collapse (min) labor_earnings, by(age)
// collapse (max) log_labor_earnings if emp == 1 & log_labor_earnings != . & labor_earnings > 0, by(age)
collapse (max) log_labor_earnings if `if_cond', by(age)
list log_labor_earnings
restore

preserve
// collapse (min) labor_earnings, by(age)
// collapse (max) labor_earnings if emp == 1 & log_labor_earnings != . & labor_earnings > 0, by(age)
collapse (max) labor_earnings if `if_cond', by(age)
list labor_earnings
restore


