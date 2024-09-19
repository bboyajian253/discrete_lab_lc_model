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
	use `dir'/UKHLS_sample, clear
	

	
	*set individual and year fixed effects also set a reference year
	xtset indiv_id year
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
	
	local minwage = 1000.00 /*minimum wage per month in UK maybe?*/
	
	
	
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
	
	******************************************************
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
   keep if age   >= 17
   
   drop if age == .
   keep if high_qual == 3 | high_qual == 4                  /* keep only GCSE and A-Level educated */ 
   sum wght
  
   drop if mental_health ==.
   drop if physical_health ==.
   sum wght
   keep if sex == 1                    /* keep only male */
	
	sum wght

   /*
   
   tab ageint10 wave    if wave<=1996
   tab ageint10 cohort  if wave<=1996
   tab wave cohort      if wave<=1996
   tab ageint10 wave    if wave==1994 | wave>=1999
   tab ageint10 cohort  if wave==1994 | wave>=1999
   tab wave cohort      if wave==1994 | wave>=1999
   */
   tab ageint10 wave    if wghtFE>0   & wghtFE !=.
   
****IDK WHAT THE LINES BELOW DO EXACTLY****
   fvset base 1946 cohort
   fvset base `refyear' wave
****IDK WHAT THE LINES ABOVE DO EXACTLY****
   
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
      
	save UKHLS_sample_p1, replace

	
	**********************************************************
	***** Construct duration dependence of health status *****
	**********************************************************
	
	use UKHLS_sample_p1, clear
	
	
	recode mh_quintiles (1=5) (2=4) (3=3) (4=2) (5=1)
	label define mh_quintiles_lab 1 "Excellent" 2 "very good" 3 "good" 4 "fair" 5 "poor" 
	label values  mh_quintiles  mh_quintiles_lab
	
	
	gen healthH5 = mh_quintiles /* THIS IS KIND OF  A BIG ASSUMPTION*/
	
	*Probably should change this to terciles for H3 and Quartiles for H4
	
	* define current healthH3
	gen healthH3 = healthH5
	recode healthH3 (1/3=1) (4=2) (5=3)
	label define healthH3lab 1 "good" 2 "fair" 3 "poor"
	label values healthH3 healthH3lab
	* define current healthH4
	gen healthH4 = healthH5
	recode healthH4 (1/2=1) (3=2) (4=3) (5=4)
	label define healthH4lab 1 "very good" 2 "good" 3 "fair" 4 "poor"
	label values healthH4 healthH4lab
	gen healthH = healthH5
	recode healthH (1/3=0) (4/5=1) 
	label define healthHlab 0 "poor" 1 "good" 
	label values healthH healthHlab 

	* adjust working, working hours, and income to be the same year as other records
*quietly {
   sort indiv_id year
        /* next 2 period health status */
   gen BadhealthH    = .
   replace BadhealthH= 0   if healthH==1
   replace BadhealthH= 1   if healthH==0
   fvset base 0 healthH

   fvset base 20 age /* this was commented out*/
   
   
   * define individual tag
   sort indiv_id year
   by indiv_id: gen IDtag = 1 if _n==1    
*} 
   
	
	quietly {
	   * define sample being observed for 2-7 consecutive periods
		gen temp = 1   if healthH !=.
		
		sort indiv_id year 
		
		gen TR2 = .
		replace TR2 = 1 if F1.temp==1 & temp==1
		gen TR3 = .
		replace TR3 = 1 if F1.temp==1 & L1.temp==1 & temp==1
		gen TR4 = .
		replace TR4 = 1 if F1.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
		gen TR5 = .
		replace TR5 = 1 if F1.temp==1 & L3.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
		gen TR6 = .
		replace TR6 = 1 if F1.temp==1 & L4.temp==1 & L3.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
		gen TR7 = .
		replace TR7 = 1 if F1.temp==1 & L5.temp==1 & L4.temp==1 & L3.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
		drop temp   
 
		* construct hazard rate
		gen     hazrB1 = 0 if TR2==1 & healthH==0
		replace hazrB1 = 1 if TR2==1 & healthH==0 & F1.healthH==1
		gen     hazrB2 = 0 if TR3==1 & healthH==0 & L1.healthH==0
		replace hazrB2 = 1 if TR3==1 & healthH==0 & L1.healthH==0 & F1.healthH==1
		gen     hazrB3 = 0 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0
		replace hazrB3 = 1 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & F1.healthH==1
		gen     hazrB4 = 0 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0
		replace hazrB4 = 1 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & F1.healthH==1
		gen     hazrB5 = 0 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0
		replace hazrB5 = 1 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & F1.healthH==1
		gen     hazrB6 = 0 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0
		replace hazrB6 = 1 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 & F1.healthH==1
		gen     hazrH1 = 0 if TR2==1 & healthH==1
		replace hazrH1 = 1 if TR2==1 & healthH==1 & F1.healthH==0
		gen     hazrH2 = 0 if TR3==1 & healthH==1 & L1.healthH==1
		replace hazrH2 = 1 if TR3==1 & healthH==1 & L1.healthH==1 & F1.healthH==0
		gen     hazrH3 = 0 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1
		replace hazrH3 = 1 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & F1.healthH==0
		gen     hazrH4 = 0 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1
		replace hazrH4 = 1 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & F1.healthH==0
		gen     hazrH5 = 0 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1
		replace hazrH5 = 1 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & F1.healthH==0   
		gen     hazrH6 = 0 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 
		replace hazrH6 = 1 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 & F1.healthH==0   
   }
   
	* fraction of fair and poor health
	tab age healthH3 [aweight=wght0] if age<=95, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure1") modify	
	putexcel  B1 = "Figure 1 (left) : Fraction of people in bad health"
	putexcel  A2 = "age"
	putexcel  A3 = matrix(yy)  B2 = "good" C2 = "fair" D2 ="poor"  
	putexcel  B3 = matrix(xx)
	
	* 1-years health transition	
	gen TrB2G = cond(healthH==0 & F1.healthH==1,1, cond(healthH==0 & F1.healthH==0,0,.))	
	gen TrG2B = cond(healthH==1 & F1.healthH==0,1, cond(healthH==1 & F1.healthH==1,0,.))  
	
	tab age TrB2G [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)	
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure1") modify	
	putexcel  H1 = "Figure 1 (middle) : % transition from bad"
	putexcel  H2 = "age"
	putexcel  H3 = matrix(yy)  I2 = "to bad"  J2 = "to good"  
	putexcel  I3 = matrix(xx)
	
	tab age TrG2B [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)		
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure1") modify	
	putexcel  M1 = "Figure 1 (right) : % transition from good"
	putexcel  M2 = "age"
	putexcel  M3 = matrix(yy)  N2 = "to bad"  O2 = "to good" 
	putexcel  N3 = matrix(xx)
	
	* duration profile for health status
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure2") modify
	xtable ageint20 [aweight=wghtFE], c(mean hazrB1 mean hazrB2 mean hazrB3 mean hazrB4 mean hazrB5)  noput
	putexcel A1 = "Figure 2 : Percentage of transitions from bad to good health conditional on being in bad health (UKHLS)"
	putexcel A2 = matrix(r(xtable)), names
	putexcel B2 = ">=1"   C2 = ">=2"    D2 = ">=3"    E2 = ">=4"    F2 = ">=5"

	xtable ageint20 [aweight=wghtFE], c(mean hazrH1 mean hazrH2 mean hazrH3 mean hazrH4 mean hazrH5)   noput  
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad health conditional on being in good health (UKHLS)"
	putexcel A12 = matrix(r(xtable)), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
	
	
	/* from here down still needs to be worked out. starts at about line 456 in the De Nardi rep code.*/
	
	* Compute number of unhealthy years conditional on being healthy/unhealthy at 55
	sort indiv_id year
	*quietly {
		preserve
		
		/*
		* adjust to 2 years age interval & keep only 2 year interval
		gen ageint2_1 =.
		forvalue ii = 55(2)69 {
			local jj = `ii'+1
			replace ageint2_1 = `ii' if agecur==`ii' | agecur==`jj'
		}
		gen     tag1 = 1 if wave==1997 & ageint2_1==agecur
		replace tag1 = 2 if wave==1997 & ageint2_1!=agecur
		bysort  headID1(tag1) : gen tag2 = tag1[1]
		sort    headID1 wave
		drop    if tag2==1 & ageint2_1!=agecur & wave<1997
		drop    if tag2==2 & ageint2_1==agecur & wave<1997
		drop    tag*
		drop    if wave<1997 & ageint2_1==F.ageint2_1  // drop those whose agecur is between 2 year interval
		
		noisily display("adjust age to 2 years interval")
		noisily count if healthH!=.
		*/
		
		* get balance panel
		keep if age>=55 & age<=65		
		egen nonmissinghealth = count(healthH==.), by(indiv_id)
		keep if nonmissinghealth==6 /*why 6??*/
		count
		sort indiv_id year
		bysort indiv_id : gen  health55 = healthH[1]   // health at 55
		keep if age>=56 & age<=65
		bysort indiv_id : egen tempH55 = total(BadhealthH) if health55==1   // unhealthy year if healthy at 55
		bysort indiv_id : egen tempU55 = total(BadhealthH) if health55==0   // unhealthy year if unhealthy at 55
		gen UHyrs5765 = cond(health55==1,tempH55, cond(health55==0,tempU55,.))
		keep if age==56
		
		putexcel set "$outdir/UKHLS_Figures", sheet("Figure5") modify
		noisily tab health55 UHyrs5765 [aweight=wghtFE], matcell(xx) matrow(yy) matcol(zz)
		mata : xx = st_matrix("xx")
		mata : xx = xx :/rowsum(xx)
		mata : st_matrix("xx",xx)
		putexcel  B1 = "No. of periods in bad health (aged 56-65)"
		putexcel  A2 = "health at 55"
		putexcel  A3 = matrix(yy)  B2 = matrix(zz)  B3 = matrix(xx)
		
		restore 
	*}
	
	
	
	save UKHLS_sample_p1, replace
	
		
	*******************************************************
	*******************************************************
	*****     CONSTRUCT CROSS SECTIONAL PROFILES      *****
	*****   i) use sampleselection1  ii) use wght0    *****
	*******************************************************
	*******************************************************
	use UKHLS_sample_p1, clear
	
	keep if sampleselection1 == 1
	drop wght wghtFE
	xtset indiv_id year
	
	**************************************************
	***** construct employment profile by health *****
	**************************************************     
	logit emp i.healthH3#i.ageint5  i.year [iweight=wght0]  if age>=21 &  age <=70, vsquish
	
	xtable age healthH3 if age>=21 & age<=64 [aweight=wght0],  c(mean   emp) sc filename($outdir/TargetedMoments.xlsx) sheet(EmploymentProfile)      format(%9.3f) modify
	putexcel set $outdir/TargetedMoments.xlsx,  sheet(EmploymentProfile) modify
	putexcel A1 = "Targted employment moments" 
	xtable age healthH3 if age>=21 & age<=64 [aweight=wght0],  c(semean emp) sc filename($outdir/TargetedMoments.xlsx) sheet(SE_EmploymentProfile)   format(%9.3f) modify
	putexcel set $outdir/TargetedMoments.xlsx,  sheet(SE_EmploymentProfile)  modify
	putexcel A1 = "SE of targted employment moments"
	xtable age healthH3 if age>=25 & age<=64 [aweight=wght0],  c(mean   emp) sc filename($outdir/UKHLS_Figures.xlsx)    sheet(Figure10_left)   format(%9.3f) modify
	putexcel set $outdir/UKHLS_Figures.xlsx,  sheet(Figure10_left)  modify
	putexcel A1 = "Figure 10 (left) : Employment by health"
	
	**************************************************
    ***** construct % bad health by income group *****
    **************************************************    
    * construct income groups by wave
	egen incgroup = xtile(labor_income) if age>=21 & age<=66, by(year ageint10_1) nq(3) weight(wght0)
	putexcel set $outdir/UKHLS_Tables.xlsx, sheet(Table9) modify
	xtable ageint10_1 incgroup [aweight=wght0], c(mean BadhealthH) sc  format(%9.3f) noput
	putexcel A1="Table 9 : % unhealthy in each income terciles (UKHLS)"
	putexcel B2= "income terciles"
	putexcel A3=matrix(r(xtable)), names

	
	************************************************************
	************************************************************
	*****     CONSTRUCT PROFILES by FIXED PRODUCTIVTY      *****
	*****   i) use sampleselection2  ii) use wghtFE        *****
	************************************************************
	************************************************************
	use UKHLS_sample_p1, clear
	keep if sampleselection1 == 1
	drop wght
	xtset indiv_id year  

	local agemin = 17   
   
   
	********************************************
    ***** Estimate fixed effect regression *****
    ********************************************	
	xtreg 	log_labor_inc i.healthH#i.age [aweight=wghtFE] if age>=`agemin', fe vsquish       /*& wave>=1984 & wave<=2017*/      
	predict loglabincHprof_NoFE   if e(sample), xb    
	predict eit                   if e(sample), e     /* predict only in-sample */                                         
  	predict ui0, u
	
	quietly: reg ui0 i.birthyear [aweight=wghtFE]       /* remove cohort effect from fixed productivity */
	predict ui   if e(sample), res
	gen  ueit  = ui  + eit 
	gen  ueit0 = ui0 + eit
	gen  loglabincHprof_wFE = loglabincHprof_NoFE + ui
	
	* assign fixed effect for period not working
	bysort indiv_id(ui) : replace ui = ui[1] if ui[1]!=.	
	sort   indiv_id year
	
	*======== create FE tercile (using individual ui and cross-sectional weight) =======
	bysort indiv_id(year) : gen tag_2013=1 if year==2013 & wghtFE>0 & wghtFE!=.
	
	xtile  ui3 = ui  [aweight=wght0]  if tag_2013==1 , nq(3)
	tab  ui3 [aweight=wght0]  if tag_2013==1 	
	

	local xx    = "ageint2"
	local agexx = 64	
	
			 
	xtable `xx' healthH  if age>=20 & age<=`agexx' & emp==1  [aweight=wght0],  c(mean   loglabincHprof_wFE) sc filename($outdir/TargetedMoments.xlsx) sheet(LaborIncProfile)   format(%9.4f) modify	
	putexcel set "$outdir/TargetedMoments.xlsx", sheet(LaborIncProfile) modify
	putexcel A1 = "Targeted log labor income profile"
	xtable `xx' healthH  if age>=20 & age<=`agexx' & emp==1  [aweight=wght0],  c(semean loglabincHprof_wFE) sc filename($outdir/TargetedMoments.xlsx) sheet(SE_LaborInc) format(%9.4f) modify
	putexcel set "$outdir/TargetedMoments.xlsx", sheet(SE_LaborInc) modify
	putexcel A1 = "SE of targeted log labor income profile"

	
	/* this is from THEIR PAPER* i need to adjust these numbers if I am going
	to report things in some sort of base year dollars*/
	
	* adjust to 2013 price (reported in the paper)
	preserve
	local avglabinc2013 = 38648
	local avglabinc1996 = 27642
	foreach var of varlist log_labor_inc loglabincHprof_NoFE loglabincHprof_wFE {
	    replace `var' = `var' + log(`avglabinc2013'/`avglabinc1996')
	}
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure10_right") modify
	xtable ageint5_1  healthH  if age>=20 & age<=`agexx' & emp==1  [aweight=wght0],  c(mean   loglabincHprof_wFE) sc   format(%9.4f) noput		
	putexcel A1 = "Figure 10 (right) : average labor income among workers by health"
	putexcel B2=matrix(r(xtable)), names
	restore
	
	summarize ui [aweight=wghtFE] if age>=20 & age<=`agexx', detail
	gen   fegroup2 = cond(ui<r(p50),1, cond(ui>=r(p50) & ui!=.,0,.))
	xtile fegroup3 = ui [aweight=wghtFE], nq(3)
	
	tab ageint2 fegroup3 [aweight=wghtFE] if healthH==1 & age>=20 & age<=65, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure6") modify	
	putexcel  B1 = "Figure 6 (left) : % bottom terciles of ﬁxed labor productivity (good)"
	putexcel  A2 = "age"
	putexcel  A3 = matrix(yy)  B2 = "bottom 1/3" C2 = "middle 1/3" D2 ="top 1/3"  
	putexcel  B3 = matrix(xx)
	
	tab ageint2 fegroup3 [aweight=wghtFE] if healthH==0 & age>=20 & age<=65, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "$outdir/UKHLS_Figures", sheet("Figure6") modify	
	putexcel  H1 = "Figure 6 (right) : % bottom terciles of ﬁxed labor productivity (bad)"
	putexcel  G2 = "age"
	putexcel  G3 = matrix(yy)  H2 = "bottom 1/3" I2 = "middle 1/3" J2 ="top 1/3"  
	putexcel  H3 = matrix(xx)	
	
	preserve
	local xx    = "ageint2"
	local agexx = 64
	

	
	collapse (mean) ui0 ui fegroup2 [aweight=wghtFE] if age>=20 & age<=`agexx', by(`xx' healthH)
	xtable `xx' healthH if ageint2>=25, c(mean fegroup2)  filename($outdir/UKHLS_Figures.xlsx) sheet(Figure3) format(%9.4f) modify
	putexcel set "$outdir/UKHLS_Figures", sheet(Figure3) modify
	putexcel A1 = "Figure 3 : Fixed labor productivity by health"
	
	restore	

	
		*** save data for estimating health shock process ***
	preserve
	keep indiv_id year wave age ageint2 ageint5 ageint10 wghtFE wght`lastWave' wght0 interest healthH healthH3 healthH4 healthH5 ui birthyear 
	/*wealth_fam3  LiteExercisePerWeek HvyExercisePerWeek  AgeFirstSmoking smokingH cigarettH DIrcpt */ 
	sort indiv_id year
// 	save healthML_data2Yinterval_wPoorHealth_R2_2_2.dta, replace
	save UKHLS_healthshock.dta, replace
	restore	
	
	*** Save data for estimating labor productivity shock ***
	preserve
	keep indiv_id year wave age ageint2 ageint5 ageint10 wghtFE wght`lastWave' wght0 interest healthH healthH3 healthH4 healthH5 ui birthyear loglabincHprof_NoFE loglabincHprof_wFE  ueit0 ueit  eit   
	/*DIrcpt  wealth_fam3*/   
	sort indiv_id year
// 	save LabProd_wPoorHealth_R2_2_2.dta, replace
	save UKHLS_laborshock.dta, replace
	restore
	
	
