cls

clear

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files" /*do file folder directory */
global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/my_model_2/input" /*output file folder directory */
global datadir  "$curdir/Data"

local dir = "$datadir"

cd "`dir'"
use "`dir'/UKHLS_sample_clean", clear

* Collapse moments by age for ages 25-75 and save the results
* employmemnt emp
preserve
collapse (mean) emp if age >= 25 & age <= 75, by(age)
tempfile emp_rate_data
save `emp_rate_data', replace
restore

preserve
collapse (sd) emp if age >= 25 & age <= 75, by(age)
rename emp sd_emp
tempfile emp_rate_sd_data
save `emp_rate_sd_data', replace
restore

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

sort age
quietly sum wage if age == 25
local initial_sd_wage = r(sd)

* Display the value of initial_sd_wage
display "Initial SD of wage for age 25: `initial_sd_wage'"

* Collapse moments by age for ages 25-75 and save the results
preserve
collapse (mean) job_hours_decimal if age >= 25 & age <= 75, by(age)
tempfile job_hours_data
save `job_hours_data', replace
restore

* Collapse wage and log_wage by age for ages 25-75 and save the results
preserve
collapse (mean) wage if age >= 25 & age <= 75, by(age)
tempfile wages_data
save `wages_data', replace
restore

* Collapse wage to sd by age for ages 25-75 and save the results
preserve
collapse (sd) wage if age >= 25 & age <= 75, by(age)
rename wage sd_wage
tempfile wages_sd_data
save `wages_sd_data', replace
restore

* Change to output directory
cd "$outdir"

* Export the job_hours dataset to a CSV file
use `job_hours_data', clear
export delimited using "labor_moments.csv", replace

* Step 4: Export the wages dataset to a CSV file
use `wages_data', clear
export delimited using "wage_moments.csv", replace

* Read the wages dataset into memory
import delimited "wage_moments.csv", clear

* Add the standard deviation to the dataset as a new observation
* gen sd_wage = `initial_sd_wage'
merge 1:1 age using `wages_sd_data'
 
* Export the dataset to a CSV file
export delimited using "wage_moments.csv", replace

* Export the wages dataset to a CSV file
use `emp_rate_data', clear
export delimited using "emp_rate_moments.csv", replace

* Read the wages dataset into memory
import delimited "emp_rate_moments.csv", clear

* Add the standard deviation to the dataset as a new observation
* gen sd_wage = `initial_sd_wage'
merge 1:1 age using `emp_rate_sd_data'
 
* Export the dataset to a CSV file
export delimited using "emp_rate_moments.csv", replace

