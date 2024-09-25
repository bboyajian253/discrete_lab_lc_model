clear
clear matrix
clear mata
set maxvar 30000

pause on
// cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
*local withYearSaveLoc "C:\Users\Ben\My Drive\PhD\PhD Year 3\3rd Year Paper\Data"
local ukhls_data_loc  "$datadir/ukhls"
// cd "$datadir/ukhls"
cd "`ukhls_data_loc'"



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
	// cd "C:\Users\Ben\3YP_Data_Work\Data\ukhls"
	cd "`ukhls_data_loc'"
	use `letter'_indresp, clear
	***this needs to change****
	gen wave = "`letter'"
	// gen year = ``i''
	***this needs to change****
	capture renpfix `letter'_
	// cd "C:\Users\Ben\3YP_Data_Work\Data"
	cd "$datadir"
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

***SAVE  RAW MERGED DATA (NO DROPS or renames)****
***save the merged and renamed data and time sstamp it
// local today "`c(current_date)'"
// local today : subinstr local today " " "", all //Format the date string to remove spaces and special characters
// local mergeName "UKHLS_merged_`today'.dta"
local mergeName "UKHLS_merged_indresp.dta"
save "`mergeName'", replace

***rename some specific vars
*UKHLS given date of survey/interview
*Year
rename istrtdaty date_year
rename intdaty_dv date_year_dv
rename date_year_dv year

*Month
rename istrtdatm date_month
rename intdatm_dv date_month_dv
drop month //drop the month variable that represnets the month of the initial sample date
rename date_month_dv month

*Day
rename istrtdatd date_day
rename intdatd_dv date_day_dv
rename date_day_dv day

*age
rename dvage age
*birth year
gen birthyear = doby_dv

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
//local labor_income_lab "monthly labor income"
label variable labor_income "monthly labor income"

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

*urban (==1) or not (==2)?
rename urban_dv urban

sort mental_health
gen mental_health_nonmissing = mental_health if mental_health != .
xtile mh_quintiles = mental_health_nonmissing, nq(5)
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

*generate hourly wage
gen wage = .
replace wage = labor_income / (4*job_hours) if job_hours > 0 & job_hours != . & labor_income > 0 & labor_income != .
label variable wage "estimated hourly wage from labor_income and job_hours"

*generate log of hourly wage
gen log_wage = log(wage)
local wage_label : variable label wage
label variable log_wage "log of `wage_label' "

*generate weekly job_hours decimal
gen job_hours_decimal = .
label variable job_hours_decimal "job_hours converted to be between 0-1"
replace job_hours_decimal = 1.0 if job_hours >= 100 //& job_hours != .
replace job_hours_decimal = job_hours / 100 if job_hours < 100

*generate age squared and cubed
gen age2 = age*age
gen age3 = age*age*age

*generate an is paid employee dummy
gen paid_emp = (job_stat == 2)
label variable paid_emp "Dummy equal to 1 if an employee"
*generate a self employed dummy
gen self_emp = (job_stat == 1)
label variable self_emp "Dummy equal to 1 if an self employed"
*gen dummy if either self employed or an employee
gen emp = (paid_emp == 1 | self_emp == 1) 
label variable emp "Dummy equal to 1 if a paid employee or self employed"

gen educHS = .
replace educHS = 1 if (high_qual == 3 | high_qual == 4 ) // if GCSE and A-Level educated
label define educHSLab 1 "High School (A level/GCSE)" 
label values educHS educHSLab

gen educHigher = .
replace educHigher = 1 if (high_qual == 1 | high_qual == 2) //if higher educated
label define educHigherLab 1 "Higher Education" 
label values educHigher educHigherLab

gen educ = .
replace educ = 1 if educHigher == 1
replace educ = 0 if (high_qual >0 & educHigher != 1) //non college
label define educLab 0 "No College" 1 "College"
label values educ educLab
	 
	 
*keep only the vars I rename + sex (i.e. keep only the vars I use)
keep age age2 age3 job_hours* mental_health job_stat job_indus job_ft ///
	indiv_id jbhas mar_stat race wave labor_income sex urban ///
	mh_* emp self_emp log_* sf_* physical_* long_weights ///
	birthyear high_qual interest psu strata wage log_wage educ educHS educHigher educ_level ///	 
	day month year
	// date_year_dv

// replace all negative numbers with missing values
foreach var of varlist _all {
    local type : type `var'
    if "`type'" == "double" | "`type'" == "float" | "`type'" == "int" | "`type'" == "long" | "`type'" == "byte" {
        di "replacing"
        replace `var' =. if `var' < 0
    }
}

*drop duplicates by indiv_id and year
duplicates report indiv_id year //report duplicates there should be none
// pause on
// pause
// duplicates drop indiv_id year, force //drp em just in case

*************************************************   
//  construct survey weights
**************************************************
gen wght = long_weight          /* use individual longitudinal weight (excl non-sample persons) */

local lastWave = "l"               
gen wght`lastWave' = -100
replace wght`lastWave' = wght if wave == "`lastWave'"  
sort indiv_id wght`lastWave'
bysort indiv_id: replace  wght`lastWave' = wght`lastWave'[_N]   /**/
recode wght`lastWave' (-100 = .)

// set weights for cross-sectional and fixed effect calculations
gen wght0    = wght          /*use for cross-sectional calculation */
gen wghtFE   = wght`lastWave'    /*  only for fixed effect regression */

**********************************************	
// Construct variables to represent age groups 
**********************************************
gen int   ageint5_1 = 1  if age >= 20 & age <=24               
replace   ageint5_1 = 2  if age >= 25 & age <=29 
replace   ageint5_1 = 3  if age >= 30 & age <=34 
replace   ageint5_1 = 4  if age >= 35 & age <=39 
replace   ageint5_1 = 5  if age >= 40 & age <=44 
replace   ageint5_1 = 6  if age >= 45 & age <=49 
replace   ageint5_1 = 7  if age >= 50 & age <=54 
replace   ageint5_1 = 8  if age >= 55 & age <=59 
replace   ageint5_1 = 9  if age >= 60 & age <=64 
replace   ageint5_1 = 10 if age >= 65 & age <=69 
replace   ageint5_1 = 11 if age >= 70 & age <=74
replace   ageint5_1 = 12 if age >= 75 & age <=79
replace   ageint5_1 = 13 if age >= 80 & age <=84
replace   ageint5_1 = 14 if age >= 85
label variable ageint5_1 "age interval" 
label define ageintlabel5_1 1  "20-24" 2  "25-29" 3 "30-34"  4 "35-39"  5  "40-44"  6  "45-49"   ///
							7  "50-54" 8  "55-59" 9 "60-64" 10 "65-69" 11  "70-74"  12 "75-79"  13  "80-84"  14 "85+", replace                          
label value ageint5_1 ageintlabel5_1 

gen ageint10_1 = .
replace ageint10_1 = 1 if age>=25 & age <=34
replace ageint10_1 = 2 if age>=35 & age <=44
replace ageint10_1 = 3 if age>=45 & age <=54
replace ageint10_1 = 4 if age>=55 & age <=64   
label define ageint10Lab_1 1 "25-34" 2 "35-44" 3 "45-54"  4 "55-64" 
label value  ageint10_1  ageint10Lab_1

* define age group
gen ageint20 = .
replace ageint20 = 1 if age>=30 & age <=54
replace ageint20 = 2 if age>=55 & age <=69
replace ageint20 = 3 if age>=70 
label define ageint20Lab 1 "30-54" 2 "55-69" 3 "70+" 
label value  ageint20  ageint20Lab	

* define age group
gen ageint20_1 = .
replace ageint20_1 = 1 if age>=30 & age <=54
replace ageint20_1 = 2 if age>=55 & age <=59
replace ageint20_1 = 3 if age>=60 & age <=64
replace ageint20_1 = 4 if age>=65 & age <=69
replace ageint20_1 = 5 if age>=70 & age <=74
replace ageint20_1 = 6 if age>=75
label define ageint20Lab_1 1 "30-54" 2 "55-59" 3 "60-64" 4 "65-69" 5 "70-74" 6 "75+" 
label value  ageint20_1  ageint20Lab_1 


***SAVE MERGED AND RENAMED (MY VARS ONLY)****	 
local myName "UKHLS_clean.dta"
save "`myName'", replace

* Erase helper files
foreach letter in `letters'{
	erase with_`letter'_stata.dta
}

