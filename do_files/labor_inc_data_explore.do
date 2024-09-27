
local if_cond = "emp == 1 & log_labor_inc != . & labor_income > 1"

preserve
// collapse (min) labor_income, by(age)
// collapse (min) log_labor_inc if emp == 1 & log_labor_inc != . & labor_income > 0, by(age)
collapse (min) log_labor_inc if `if_cond', by(age)
list log_labor_inc
restore

preserve
// collapse (min) labor_income, by(age)
// collapse (min) labor_income if emp == 1 & log_labor_inc != . & labor_income > 0, by(age)
collapse (min) labor_income if `if_cond', by(age)
list labor_income
restore

preserve
// collapse (min) labor_income, by(age)
// collapse (max) log_labor_inc if emp == 1 & log_labor_inc != . & labor_income > 0, by(age)
collapse (max) log_labor_inc if `if_cond', by(age)
list log_labor_inc
restore

preserve
// collapse (min) labor_income, by(age)
// collapse (max) labor_income if emp == 1 & log_labor_inc != . & labor_income > 0, by(age)
collapse (max) labor_income if `if_cond', by(age)
list labor_income
restore


