cls
clear

// global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
// global dodir    "$curdir/Do_Files" /*do file folder directory */
// global datadir  "$curdir/Data"
// global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input/" /*output file folder directory */

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

sum age emp job_hours wage labor_income if emp == 1
// xtile wage_percentiles = wage if emp == 1 & wage !=. [pweight = wght], nq(1000)  // Generate 1000 quantiles only for emp == 1
// xtile wage_percentiles = wage if emp == 1 & wage !=., nq(1000)  // Generate 1000 quantiles only for emp == 1

// drop if wage_percentiles > 998 & emp == 1 & wage !=.  // Drop the top 1% of wage earners
// do "$dodir/test.do"

drop if emp == 1 & wage < 5.22 // minimum wage in the uk is 11.44 euros for those over 21 years of age
// drop if emp == 1 & job_hours < 5.0
drop if emp == 1 & job_hours < 10.0
// do "$dodir/test.do"
// do "$dodir/duplicates_id_age.do"
// pause on
// pause

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

// pause on
// pause

do "$dodir/health_states_types.do"

save UKHLS_sample, replace

di "*****got to the end of UKHLS_sample.do*****"

*************************************************************************************
//  sample selection for general sample statistics  
//  i) observe labor income, ii) observe work status,
*************************************************************************************

// gen byte  sampleselection1 =.
// gen temp1 = 1 if labor_income !=. & labor_income>=0      /* observe labor income    << RR3 >>. */
// gen temp2 = 1 if emp == 0 | emp == 1                                  /* observe works status    << RR3 >>. */

// bysort indiv_id : egen labincseries = total(temp1)
// bysort indiv_id : egen workseries   = total(temp2)
// replace sampleselection1 = 1 if labincseries>=3 | workseries>=3        /* << RR3 >> */
// sort indiv_id wave
// drop temp1 temp2

// * Check sample size
// tab ageint10 wave if sampleselection1 ==1
// tab ageint10 wave if sampleselection1 ==1 & wghtFE>0 & wghtFE !=.

// *tab ageint10  mental_health  if sampleselection1 ==1
// tab ageint10  cohort   if sampleselection1 ==1
// tab ageint10  wave     if sampleselection1 ==1
// tab ageint10  high_qual    if sampleselection1 ==1
      
// ***we want to drop the never employed

// // Sort the data by indiv_id and year
// sort indiv_id year

// // Create a variable to indicate if an individual is ever employed
// capture drop ever_employed
// egen ever_employed = max(emp == 1), by(indiv_id)

// *drop if they have never been employed
// drop if ever_employed == 0

// *drop indicator variable
// drop ever_employed

// ***we want to stop following people after they become self employed
// sort indiv_id year
// *find the first year someone is self employed
// egen first_self_emp_year = min(year/ (self_emp ==1)), by(indiv_id)
// *generate a dummy for all periods after an individual has been self employed
// by indiv_id: gen has_been_self_emp = (year >= first_self_emp_year)
// *drop once an individual becomes self employed
// drop if has_been_self_emp == 1
// *drop indicator variables
// drop has_been_self* first_self*