clear
clear matrix
clear mata
set maxvar 30000

pause on
cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
*local withYearSaveLoc "C:\Users\Ben\My Drive\PhD\PhD Year 3\3rd Year Paper\Data"



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


// Define the list of years and wave letters
local years "2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020"
local letters "a b c d e f g h i j k l"

****create year variable****
*delete prefixes

local n : word count `years'
tokenize "`years'" 

forvalues i = 1/`n'{
	local letter : word `i' of `letters'
	cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
	use `letter'_indresp, clear
	***this needs to change****
	gen wave = "`letter'"
	gen year = ``i''
	***this needs to change****
	capture renpfix `letter'_
	cd "C:\Users\Ben\3YP_Data_Work\Data"
	save with_`letter'_stata, replace

}

/*
****create year variable****
*delete prefixes
foreach letter in `letters'{
		cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
		use `letter'_indresp, clear
		***this needs to change****
		gen wave = "`letter'"
		*gen year = `year'
		***this needs to change****
		capture renpfix `letter'_
		cd "C:\Users\Ben\3YP_Data_Work\Data"
		save with_`letter'_stata, replace
}
*/


use with_a_stata, clear
local myLetters "b c d e f g h i j k l"
// Loop through the years
foreach letter in `myLetters' {
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

*rename date_year_dv year

*Month
rename istrtdatm date_month
rename intdatm_dv date_month_dv

*Day
rename istrtdatd date_day
rename intdatd_dv date_day_dv

*age
rename dvage age
*birth year
gen birthyear = year - age

*interest/dividends proxy for wealth
rename fiyrdia interest //amount of interest and dividends
rename fiyrdb1 int_over1 //greater than 500 pounds in interest and divs
rename fiyrdb2 int_over2 // 1000
rename fiyrdb3 int_over3 // 5000
rename fiyrdb4 int_over4 // 2500
rename fiyrdb5 int_over5 // 10000
rename fiyrdb6 int_over6  //0ver 100
 
*no. of hours normally worked per week 
rename jbhrs job_hours
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
rename scsf1 sf_gen_health 
*health limits typical activities
rename scsf2a sf_limits_act
*health limits climbing stairs
rename scsf2b sf_limits_stairs
*health limits work
rename scsf3a sf_limits_work
*health limits kind of work
rename scsf3b sf_limits_work_type
*emotional problems caused you to accomplish less
rename scsf4a  sf_mental_accomplish
*emotional problems cause to be less careful/more risky?
rename scsf4b sf_mental_careful
*pain interferes with work
rename scsf5 sf_pain_work
*felt calm and peaceful
rename scsf6a sf_calm
*had a lot of energy
rename scsf6b sf_energy
*felt downhearted and depressed
rename scsf6c sf_felt_depressed
*health or emotional problems interfered with social activities
rename scsf7 sf_limit_social
*scsf1 is teh self-completion version of the question. in wave = a which is the first wave there was no self-completion portion so you need to use the sf1 question from interviews.
if wave == "a"{
	*each of the health (physical and mental) shortform questions
	*general health
	sf_gen_health = sf1 
	*health limits typical activities
	sf_limits_act = sf2a 
	*health limits climbing stairs
	sf_limits_stairs = sf2b 
	*health limits work
	sf_limits_work = sf3a 
	*health limits kind of work
	sf_limits_work_type = sf3b 
	*emotional problems caused you to accomplish less
	sf_mental_accomplish = sf4a  
	*emotional problems cause to be less careful/more risky?
	sf_mental_careful = sf4b 
	*pain interferes with work
	sf_pain_work = sf5 
	*felt calm and peaceful
	sf_calm = sf6a 
	*had a lot of energy
	sf_energy = sf6b 
	*felt downhearted and depressed
	sf_felt_depressed = sf6c 
	*health or emotional problems interfered with social activities
	sf_limit_social = sf7 

}


*rename longitudinal survey weights
rename indinus_lw long_weights

*job_status: Current economic activity
rename jbstat job_stat
*job industry: Current job: Industrial classification (CNEF), two digits 
rename jbiindb_dv job_indus
*job full time or part time unemployment
rename jbft_dv job_ft
*individual identifier across years
rename pidp indiv_id
*derived income vairables
rename fimnnet_dv inc_net_month
rename fimngrs_dv inc_gross_month
rename fimngrs_if inc_if
rename fimnlabnet_dv labor_income

*set a reference year

	local refyear = 2015
	local baseyear = 2008
	
	*UK CPI all items from FRED. Ref year: 2015	
	local cpi2008 = 86.2
	local cpi2009 = 87.9
	local cpi2010 = 90.1
	local cpi2011 = 93.6
	local cpi2012 = 96.0
	local cpi2013 = 98.2
	local cpi2014 = 99.6
	local cpi2015 = 100.0
	local cpi2016 = 101.0
	local cpi2017 = 103.6
	local cpi2018 = 106.0 
	local cpi2019 = 107.8 
	local cpi2020 = 108.9  	
	local cpi2021 = 111.6  
	
	local minwage = 1700.00  /*minimum wage per month in UK maybe?*/
	
	
	
	******************************************
	**** average labor income of baseyear ****
	******************************************
	
	***BAD MAYBE: Keep missing labor income
	
	local cpiRatio = `cpi`refyear'' / `cpi`baseyear''
	* Generate the variable only for non-missing values of labor_income
	gen totlabincHcur`refyear' = .
	replace totlabincHcur`refyear' = labor_income * `cpiRatio' if !missing(labor_income)

	* Summarize the new variable
	summarize totlabincHcur`refyear' if year==`refyear', detail
	drop totlabincHcur`refyear'
	
	summarize labor_income if inlist(year, 2014, 2015, 2016) & labor_income >= `minwage', detail
	
	
*education (highest qualification)
rename qfhigh educ_level_curr //current reported education level detailed education categories
rename qfhigh_dv educ_level //highest ever reported education level detailed education categories
rename hiqual_dv high_qual //highest ever reported qualification, less detailed, includes a category 'other' that might include trade certificates? Idk?

*marriage status (harmonized and derived/de facto)
rename marstat_dv mar_stat

*race (derived from first response to the question)
rename racel_dv race

*government office region 
rename gor_dv gor

*country of residence (name is the same)
*sum country

*urban (==1) or not (==2)?
rename urban_dv urban


sort mental_health
// Generate MH quintiles using xtile
// Exclude missing values from 'interest' and create a new variable 'interest_nonmissing'
gen mental_health_nonmissing = mental_health if mental_health != .

// Create quintiles based on 'interest_nonmissing'
xtile mh_quintiles = mental_health_nonmissing, nq(5)

// Drop the temporary variable 'interest_nonmissing'
drop mental_health_nonmissing

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


*generate log of labor income
gen log_labor_inc = log(labor_income)
local inc_label : variable label labor_income
label variable log_labor_inc "log of `inc_label' "


*generate log of gross income
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
/*
*keep only the vars I rename + sex (i.e. keep only the vars I use)
keep age* job_hours job2_hours job2_has mental_health* job_stat job_indus job_ft ///
	 indiv_id jbhas educ_level* mar_stat race gor country date_* wave labor_income ///
	 inc* sex urban mh* emp* paid* self* log_* sf_* physical_* year long_weights ///
	 birthyear high_qual interest ///
*/

pause
	 
*keep only the vars I rename + sex (i.e. keep only the vars I use)
keep age age2 age3 job_hours mental_health job_stat job_indus job_ft scghq* ///
	 indiv_id jbhas educ_level mar_stat race wave labor_income ///
	 sex urban mh_* emp self_emp log_* sf_* physical_* year long_weights ///
	 birthyear high_qual interest psu strata ///	 
	 
// replace all negative numbers with missing values
foreach var of varlist _all {
    local type : type `var'
    if "`type'" == "double" | "`type'" == "float" | "`type'" == "int" | "`type'" == "long" | "`type'" == "byte" {
        di "replacing"
        replace `var' =. if `var' < 0
    }
}

	 
*drop duplicates by indiv_id and year
duplicates report indiv_id year /*what do the duplicates look like*/
duplicates drop indiv_id year, force /*drop 'em*/

	 
***SAVE MERGED AND RENAMED (MY VARS ONLY)****	 
* Construct the filename with a timestamp
local myName "UKHLS_sample_clean.dta"
* Save the cleaned data with the timestamped filename
save "`myName'", replace

* Erase helper files
foreach letter in `letters'{
	erase with_`letter'_stata.dta
}


/*
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



