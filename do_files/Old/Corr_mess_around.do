// Step 1: Run simple regressions
reg labor_income mental_health
reg labor_income physical_health

// Step 2: Run multiple regression
reg labor_income mental_health physical_health

// Step 3: Calculate indirect effect
di "Indirect effect of mental_health on labor_income through physical_health: " ///

nlcom (indirect_effect: _b[mental_health] * _b[physical_health])


// Step 4: Check for multicollinearity
cor mental_health physical_health
