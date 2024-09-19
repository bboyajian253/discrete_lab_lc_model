***BASED ON CODE WITH THE FOLLOWING HEADING***
   * This code compute several statistics from PSID and data for later used to estimate labor productivty shock and health shock process.
   *    updated: Dec 1, 2023   
   
   *  Required input files (stored in 'curdir')
   *    - PSIDsample.dta (automatically created from IND2017ER_rev1.do)
   *
   *  Output files 
   *    (saved in 'curdir')
   *    - PSID_healthshock.dta for estimating health shock parameters
   *    - PSID_laborshock.dta for estimating labor productivity parameters
   *    (saved in 'outdir')
   *    - PSID_Figures.xlsx for lated used in Matlab code to plot figures in the paper and to generate input files from lifecycle model (xxx.in) 
   *    - PSID_Tables.xlsx containing estimates reported in the papers 
   *    - TargetedMoment.xlsx containing estimates later used as targeted moments in life cycle model.

	
	cls
	clear
     
	local dir = "$datadir"
 
	
	cd  `dir'	
	use `dir'/UKHLS_sample_clean, clear
	

	
	*set individual and year fixed effects
	xtset indiv_id year
	
	 * define age group
   gen ageint2 = .
   local agex = 21
   forvalue i = `agex'(2)99 {       
      replace ageint2 = `i' if age>=`i' & age<=`i'+1
   }	     
   label define ageint2Lab 21 "21-22" 23 "23-24" 25 "25-26" 27 "27-28" 29 "29-30" 31 "31-32" 33 "33-34" 35 "35-36" 37 "37-38" 39 "39-40" ///
                           41 "41-42" 43 "43-44" 45 "45-46" 47 "47-48" 49 "49-50" 51 "51-52" 53 "53-54" 55 "55-56" 57 "57-58" 59 "59-60" ///
						   61 "61-62" 63 "63-64" 65 "65-66" 67 "67-68" 69 "69-70" 71 "71-72" 73 "73-74" 75 "75-76" 77 "77-78" 79 "79-80" ///
						   81 "81-82" 83 "83-84" 85 "85-86" 87 "87-88" 89 "89-90" 91 "91-92" 93 "93-94" 95 "95-96" 97 "97-98" 99 "99+"
   label value  ageint2  ageint2Lab
   macro drop agex
   
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
   
     
   ****************************************************************
   *****           Generate 5 year cohort variable            *****
   ***** cohorts of all people older than or equal 21 in 1984 *****
   ****************************************************************
quietly {
   local cohortyear = 5
   local cohort1    = 1911   /* at least 72 in 1984 */
   local cohortN    = 1999   /* 21-26 in 2017  */ 
   local cohortNp   = 2001   /* 16-20 in 2017  */
   gen  cohort = .                      
   replace cohort = `cohort1'  if birthyear <= 1913                      /* at least 71 in 1984  */
   replace cohort = `cohortN'  if birthyear >= 1994 & birthyear <=1998   /* 21-26 in 1984  */
   replace cohort = `cohortNp' if birthyear >= 1999 & birthyear <=2003   /* 16-20 in 1984  */
   forvalue yy = 1914(`cohortyear')1989 {    
       replace cohort = (`yy'+`yy'+`cohortyear'-1)/2  if birthyear >=`yy' & birthyear <`yy'+`cohortyear'
   }
   * additional young cohort
   replace cohort = `cohortNp'  if birthyear >= `cohortNp'-2                          
}
	
*** construct weight for the following calculation ***
	******************************************************
	
	
*quietly {
   gen wght = long_weight          /* use individual longitudinal weight (excl non-sample persons) */
   
   local lastWave = "l"               
   gen wght`lastWave' = -100
   replace wght`lastWave' = wght if wave == "`lastWave'"  
   sort indiv_id wght`lastWave'
   bysort indiv_id: replace  wght`lastWave' = wght`lastWave'[_N]   /**/
   recode wght`lastWave' (-100 = .)
 
   
   * choose weight for calculation
   gen wght0    = wght          /*  use for cross-sectional calculation */
   gen wghtFE   = wght`lastWave'    /*  only for fixed effect regression */
*}
   
   ****************************************************************
   *****            GENERAL SAMPLE SELECTION                  *****
   ***** 1) positive weight, 2) observe age and at least 21   *****
   ***** 3) observe health status 4) only highschool          *****
   ****************************************************************
	sum wght
   
	drop if wght ==. | wght<=0              /* drop those without weight */   
	*keep if age   >= 17 & cohort !=.     /* keep only cohort born before or in 1964 (at least age 21 in 1984)  */
	*****this was changed*****
	keep if age   >= 17
   ******see above*******
   
	drop if age == .
	gen educHS = .
	replace educHS = 1 if (high_qual == 3 | high_qual == 4 ) // if GCSE and A-Level educated
	label define educHSLab 1 "High School (A level/GCSE)" 
	label values educHS educHSLab
	
	gen educHigher = .
	replace educHigher = 1 if (high_qual == 1 | high_qual == 2) //if higher educated
	*drop otherwise
	drop if (high_qual <0) // this keeps everyone who reports an answer to high_qual
	label define educHigherLab 1 "Higher Education" 
	label values educHigher educHigherLab
	
	gen educ = .
	replace educ = 1 if educHigher == 1
	replace educ = 0 if (high_qual >0 & educHigher != 1) //non college
	label define educLab 0 "No College" 1 "College"
	label values educ educLab
	
	*drop if educ == 1
	
	*drop if labor_income == 0 & emp == 1 & job_hours == . //drop if descripency in employment and hours worked about 2k observations
	
	sum wght
  
	drop if mental_health ==.
	drop if physical_health ==.
	sum wght
	*keep if sex == 1                    /* keep only male */
	
	sum wght
	
	   
      *************************************************************************************
   ***** sample selection for all statistics except those related to assets (1)    *****
   *****                 (use wght0 in calculation)                                *****
   ***** i) observe labor income, ii) observe work status, iii) wave 1984-1996     *****                                               
   *************************************************************************************
   gen byte  sampleselection1 =.
   gen temp1 = 1 if labor_income !=. & labor_income>=0      /* observe labor income    << RR3 >>. */
   gen temp2 = 1 if emp == 0 | emp == 1                                  /* observe works status    << RR3 >>. */
	
	bysort indiv_id : egen labincseries = total(temp1)
	bysort indiv_id : egen workseries   = total(temp2)
	replace sampleselection1 = 1 if labincseries>=3 | workseries>=3        /* << RR3 >> */
	sort indiv_id wave
	drop temp1 temp2

   
	* Check sample size
	tab ageint10 wave if sampleselection1 ==1
	tab ageint10 wave if sampleselection1 ==1 & wghtFE>0 & wghtFE !=.

   
	*tab ageint10  mental_health  if sampleselection1 ==1
	tab ageint10  cohort   if sampleselection1 ==1
	tab ageint10  wave     if sampleselection1 ==1
	tab ageint10  high_qual    if sampleselection1 ==1
      
	save UKHLS_sample, replace
	
	
