clear
clear matrix
clear mata
set maxvar 30000

cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
*local withYearSaveLoc "C:\Users\Ben\My Drive\PhD\PhD Year 3\3rd Year Paper\Data"

local letters "a b c d e f g h i j k l"

local myData2009 "a_indresp"
local myData2010 "b_indresp"
local myData2011 "c_indresp"
local myData2012 "d_indresp"
local myData2013 "e_indresp"
local myData2014 "f_indresp"
local myData2015 "g_indresp"
local myData2016 "h_indresp"
local myData2017 "i_indresp"
local myData2018 "j_indresp"
local myData2019 "k_indresp"
local myData2020 "l_indresp"


// Define the list of years
local years "2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020"

****create year variable****
*delete prefixes
foreach letter in `letters'{
	foreach year in `years'{
		cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
		use `letter'_indresp, clear
		***this needs to change****
		gen wave = "`letter'"
		gen year = `year'
		***this needs to change****
		capture renpfix `letter'_
		cd "C:\Users\Ben\3YP_Data_Work\Data"
		save with_`letter'_stata, replace
	}
}

use with_a_stata, clear

// Loop through the years
foreach letter in `letters' {
	di " ***** joining for wave: `letter' *****"
    // Specify the current dataset
    append using with_`letter'_stata
}
*
sort pidp




***rename some specific vars
*UKHLS given date of survey/interview
*Year
rename istrtdaty date_year
rename intdaty_dv date_year_dv

*Month
rename istrtdatm date_month
rename intdatm_dv date_month_dv

*Day
rename istrtdatd date_day
rename intdatd_dv date_day_dv

*age
capture rename dvage age 
*no. of hours normally worked per week 
capture rename jbhrs job_hours
*no. of hours normally worked per month 2nd job 
rename j2hrs job2_hours
*has a second job
rename j2has job2_has

*mental health index SF-12
rename sf12m* mental_health
*physical health index from SF-12
rename sf12p* physical_health

*each of the health (physical and mental) shortform questions
*general health
rename sf1 sf_gen_heatlh 
*health limits typical activities
rename sf2a sf_limits_act
*health limits climbing stairs
rename sf2b sf_limits_stairs
*health limits work
rename sf3a sf_limits_work
*health limits kind of work
rename sf3b sf_limits_work_type
*emotional problems caused you to accomplish less
rename sf4a  sf_mental_accomplish
*emotional problems cause to be less careful/more risky?
rename sf4b sf_mental_careful
*pain interferes with work
rename sf5 sf_pain_work
*felt calm and peaceful
rename sf6a sf_calm
*had a lot of energy
rename sf6b sf_energy
*felt downhearted and depressed
rename sf6c sf_felt_depressed
*health or emotional problems interfered with social activities
rename sf7 sf_limit_social

*rename longitudinal survey weights


*job_status: Current economic activity
capture rename jbstat job_stat
*job industry: Current job: Industrial classification (CNEF), two digits 
capture rename jbiindb_dv job_indus
*job full time or part time unemployment
capture rename jbft_dv job_ft
*individual identifier across years
capture rename pidp indiv_id
*derived income vairables
rename fimnnet_dv inc_net_month
rename fimngrs_dv inc_gross_month
rename fimngrs_if inc_if
rename fimnlabnet_dv labor_income

*education (highest qualification)
rename qfhigh educ_level_curr
rename qfhigh_dv educ_level

*marriage status (harmonized and derived/de facto)
rename marstat_dv mar_stat

*race (derived from first response to the question)
rename racel_dv race

*government office region 
rename gor_dv gor

*country of residence (name is the same)
sum country

*urban (==1) or not (==2)?
rename urban_dv urban

drop if mental_health <= 0
capture drop mh_q*

// Sort the data in ascending order
sort mental_health

// Generate quintiles using xtile
xtile mh_quintiles = mental_health, nq(5)

// Generate dummy variables for quintiles
gen mh_q1 = (mh_quintiles == 1)
gen mh_q2 = (mh_quintiles == 2)
gen mh_q3 = (mh_quintiles == 3)
gen mh_q4 = (mh_quintiles == 4)
gen mh_q5 = (mh_quintiles == 5)

// Label the dummy variables
label variable mh_q1 "Poor Mental Health: Quintile 1"
label variable mh_q2 "Fair Mental Health: Quintile 2"
label variable mh_q3 "Good Mental Health: Quintile 3"
label variable mh_q4 "Very Good Mental Health: Quintile 4"
label variable mh_q5 "Excellent Mental Health: Quintile 5"

// Display table of statistics for each quintile 
tabstat mental_health, stat(n mean min max sd p50) by(mh_quintiles)

*generate log of income
gen log_gross_inc = log(inc_gross_month)
local inc_label : variable label inc_gross_month
label variable log_gross_inc "log of `inc_label' "

*generate log of hours
gen log_hours = log(job_hours)
local inc_label : variable label job_hours
label variable log_hours "log of `inc_label' "


*generate age squared and cubed
gen age2 = age*age
gen age3 = age*age*age

*generate an is paid employee dummy
gen paid_emp = (job_stat == 2)
label variable paid_emp "Dummy equal to 1 if an employee"
*generate a self employed dummy
gen self_emp = (job_stat == 1)
label variable self_emp "Dummy equal to 1 if an self employed"
*gen dummt if either self employed or an employee
gen emp = (paid_emp == 1 | self_emp == 1) 
label variable emp "Dummy equal to 1 if a paid employee or self employed"

***SAVE MERGED AND RENAMED (NO DROPS)****
***save the merged and renamed data and time sstamp it
* Get the current date
local today "`c(current_date)'"

* Format the date string to remove spaces and special characters
local today : subinstr local today " " "", all

* Construct the filename with a timestamp
local mergeName "UKHLS_merged_`today'.dta"
* Save the cleaned data with the timestamped filename
*save "`mergeName'", replace
*###UNCOMMENT THE ABOVE LINE TO SAVE "RAW" MERGED DATA###*

***start to clean it up a bit

*keep only the vars I rename + sex (i.e. keep only the vars I use)
keep age* job_hours job2_hours job2_has mental_health* job_stat job_indus job_ft ///
	 indiv_id jbhas educ_level* mar_stat race gor country date_* ///
	 inc* sex urban mh* emp* paid* self* log_* sf_* physical_* year ///
	 
	 
***SAVE MERGED AND RENAMED (MY VARS ONLY)****	 
* Construct the filename with a timestamp
local myName "UKHLS_sample.dta"
* Save the cleaned data with the timestamped filename
save "`myName'", replace



***drop non-Men
drop if sex != 1	 

***drop coded value for "proxy, inapplicable, dont know, refusal etc."
*non-self employed avg hrs per week
*drop if working_hours <= -1
*mental health measure
drop if mental_health <= -1 
*drop if job_indus <=-1
drop if job_stat <=-1


*** drop if not working age (roughly: we extend it a little bit to observe a longer trend over the lifecycle)
drop if age > 70
drop if age < 16

***we want to drop the never employed

// Sort the data by indiv_id and year
sort indiv_id year

// Create a variable to indicate if an individual is ever employed
capture drop ever_employed
egen ever_employed = max(emp == 1), by(indiv_id)

*drop if they have never been employed
drop if ever_employed == 0

*drop indicator variable
drop ever_employed

***we want to stop following people after they become self employed
sort indiv_id year
*find the first year someone is self employed
egen first_self_emp_year = min(year/ (self_emp ==1)), by(indiv_id)
*generate a dummy for all periods after an individual has been self employed
by indiv_id: gen has_been_self_emp = (year >= first_self_emp_year)
*drop once an individual becomes self employed
drop if has_been_self_emp == 1
*drop indicator variables
drop has_been_self* first_self*


***SAVE MERGED AND RENAMED MY VARS ONLY, DROP BAD OBSERVATIONS***

*save the cleaned data and time sstamp it
* Get the current date
local today "`c(current_date)'"
* Format the date string to remove spaces and special characters
local today : subinstr local today " " "", all
* Construct the filename with a timestamp
local cleanName "UKHLS_my_vars_cleaned_`today'.dta"
* Save the cleaned data with the timestamped filename
save "`cleanName'", replace


* Erase helper files
foreach letter in `letters'{
	erase with_`letter'_stata.dta
}

