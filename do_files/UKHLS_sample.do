cls
clear

local dir = "$datadir"
cd  "`dir'"	
use "`dir'/UKHLS_clean", clear

// HERE IS A CHANGE

**************************************
// General sample selection
**************************************
drop if mental_health ==.
drop if physical_health ==.
drop if wght ==. | wght<=0              
drop if age == .
// drop if emp == 1 & (wage == . | wage <= 0) 

// xtile wage_percentiles = wage if emp == 1 & wage !=. [pweight = wght], nq(1000)  // Generate 1000 quantiles only for emp == 1
// xtile wage_percentiles = wage if emp == 1 & wage !=., nq(1000)  // Generate 1000 quantiles only for emp == 1
// drop if wage_percentiles > 998 & emp == 1 & wage !=.  // Drop the top 1% of wage earners
drop if emp == 1 & missing(job_hours)
drop if emp == 1 & missing(labor_earnings)

drop if emp == 1 & wage < 5.22 // minimum wage in the uk is 11.44 euros for those over 21 years of age
drop if emp == 1 & job_hours < 10.0

local start_age = 25
local end_age = 75
drop if age < `start_age'
drop if age > `end_age' + 1

duplicates report indiv_id age
duplicates report indiv_id year
duplicates report indiv_id year age

// DROP AFTER DUPLICATE BY indiv_id and age appears 
* First, sort the data by indiv_id and age
sort indiv_id age
* Now flag the first duplicate occurrence of age within each indiv_id
by indiv_id (age): gen byte first_dup_age = (age == age[_n-1] & indiv_id == indiv_id[_n-1])
* Create a cumulative sum to identify all observations after the first duplicate
by indiv_id: gen byte drop_obs = sum(first_dup_age)
* Keep only observations before and including the first duplicate age
keep if drop_obs == 0

duplicates report indiv_id age
duplicates report indiv_id year
duplicates report indiv_id year age


do "$dodir/health_states_types.do"

save UKHLS_sample, replace

di "*****got to the end of UKHLS_sample.do*****"