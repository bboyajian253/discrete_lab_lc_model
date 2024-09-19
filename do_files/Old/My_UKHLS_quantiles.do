**********************************************************
	***** Construct duration dependence of health status *****
	**********************************************************
	cls
	clear
     
	local dir = "$datadir"
 
	
	cd  `dir'	
	
	
	use UKHLS_sample_p1, clear
	***drop non men***
	*keep if sex==1
	
	drop mh_q*
	
	***Interest/wealth proxy calculations
	// Sort the data in ascending order
	sort interest
	gen interest_nonmissing = interest if interest != .
	//Create quintiles based on 'interest_nonmissing'
	xtile int_Q5 = interest_nonmissing, nq(5)
	// Drop the temporary variable 'interest_nonmissing'
	drop interest_nonmissing
	sum mental_health, detail
	return list
	gen median_mh = (mental_health >  r(p50))
	sum median*
	sum physical_health, detail
	return list
	gen median_ph = (physical_health > r(p50))
	sum median*
	

	
	//make physical and mental health quantiles for some quantile q
	local quants = "2 3 4 5 8 10"
	foreach q in `quants'{
		sort physical_health
		gen physical_health_nonmissing = physical_health if physical_health != .
		xtile ph_Q`q' = physical_health_nonmissing, nq(`q')
		drop physical_health_nonmissing  
	
		sort mental_health
		gen mental_health_nonmissing = mental_health if mental_health != .
		xtile mh_Q`q' = mental_health_nonmissing, nq(`q')
		drop mental_health_nonmissing  // Sort the data in ascending order
	}	
	
	* Label the values of mh_Q5
	label define mh_q5lab 1 "poor mh" 2 "fair mh" 3 "good mh" 4 "very good mh" 5 "excellent mh"
	label values mh_Q5 mh_q5lab
	label define ph_q5lab 1 "poor ph" 2 "fair ph" 3 "good ph" 4 "very good ph" 5 "excellent ph"
	label values ph_Q5 ph_q5lab
	* Label the values of mh_Q4
	label define mh_q4lab 1 "poor mh" 2 "fair mh" 3 "good mh" 4 "very good mh"
	label values mh_Q4 mh_q4lab
	label define ph_q4lab 1 "poor ph" 2 "fair ph" 3 "good ph" 4 "very good ph" 
	label values ph_Q4 ph_q4lab
	* Label the values of mh_Q3
	label define mh_q3lab 1 "poor mh" 2 "fair mh" 3 "good mh" 
	label values mh_Q3 mh_q3lab
	label define ph_q3lab 1 "poor ph" 2 "fair ph" 3 "good ph" 
	label values ph_Q3 ph_q3lab
	* Label the values of mh_Q2
	label define mh_q2lab 1 "poor mh" 2 "fair mh" 
	label values mh_Q2 mh_q2lab
	label define ph_q2lab 1 "poor ph" 2 "fair ph" 
	label values ph_Q2 ph_q2lab
	*svyset psu [pweight=long_weights], strata(strata)
	xtset year
	*set fixed value comparison category
	
	local spec1 "i.mh_Q`qq' i.ph_Q5 urban i.race age age2 age3 i.mar_stat sex"	
	local deps "log_labor_inc log_hours"
	
cd "$outdir"
foreach q in `quants'{
	local qq = `q' //what quantile are we working with?
	* Run xtreg and store the results
	xtreg emp `spec1' //`weights'
	estimates store Reg_emp

	* Perform multiple coeff equality tests and store the results
	test 2.mh_Q`qq'==0
	mat m0 =(r(chi2), r(p))
	mat rowname m0="Test 0"
	local m m0
		
	if `qq' > 2 {
		test 2.mh_Q`qq'=3.mh_Q`qq'
		mat m1 =(r(chi2), r(p))
		mat rowname m1="Test 1"
		local m `m' \m2
	}
	if `qq' > 3 {	
		capture test 3.mh_Q`qq'=4.mh_Q`qq'
		mat m2 = (r(chi2), r(p))
		mat rowname m2 = "Test 2"
		local m `m' \m2
	}
	if `qq' > 4 {	
		test 4.mh_Q`qq'=5.mh_Q`qq'
		mat m3 = (r(chi2), r(p))
		mat rowname m3 = "Test 3"
		local m `m' \m3
		mat m = `m'
	}

	* Export the results to an Excel file using putexcel
	
	* Export the results to an Excel file using putexcel
	putexcel set "coeff_tests_Q`qq'.xlsx", sheet(emp) modify
	putexcel B1= "Chi2"
	putexcel C1= "P Value"
	putexcel A2 = "2.mh_Q`qq' == 0"
	putexcel A3= "2.mh_Q`qq' - 3.mh_Q`qq' = 0"
	putexcel A4= "3.mh_Q`qq' - 4.mh_Q`qq' = 0"
	putexcel A5= "4.mh_Q`qq' - 5.mh_Q`qq' = 0"
	putexcel B2= matrix(m)


	foreach dep in `deps'{

		// Run regression with year fixed effects, using mh_q1 as the excluded group (comparison group)
		xtreg `dep' `spec1' if emp == 1  // `weights'
		// Save the regression results
		estimates store Reg_`dep'
		
		test 2.mh_Q`qq'==0
		mat m0 =(r(chi2), r(p))
		mat rowname m0="Test 0"
		local m m0
	
	if `qq' > 2 {
		test 2.mh_Q`qq'=3.mh_Q`qq'
		mat m1 =(r(chi2), r(p))
		mat rowname m1="Test 1"
		local m `m' \m2
	}
	if `qq' > 3 {	
		capture test 3.mh_Q`qq'=4.mh_Q`qq'
		mat m2 = (r(chi2), r(p))
		mat rowname m2 = "Test 2"
		local m `m' \m2
	}
	if `qq' > 4 {	
		test 4.mh_Q`qq'=5.mh_Q`qq'
		mat m3 = (r(chi2), r(p))
		mat rowname m3 = "Test 3"
		local m `m' \m3
		mat m = `m'
	}	
		* Export the results to an Excel file using putexcel
		putexcel set "coeff_tests_Q`qq'.xlsx", sheet(`dep') modify
		putexcel B1= "Chi2"
		putexcel C1= "P Value"
		putexcel A2 = "2.mh_Q`qq' == 0"
		putexcel A3= "2.mh_Q`qq' - 3.mh_Q`qq' = 0"
		putexcel A4= "3.mh_Q`qq' - 4.mh_Q`qq' = 0"
		putexcel A5= "4.mh_Q`qq' - 5.mh_Q`qq' = 0"
		putexcel B2= matrix(m)

	
	}
	
***Combine and export the stored results to a text file***
*CHANGE TO THE SAVE LOCATION
	cd "$outdir"
	

	

	esttab Reg_emp Reg_log_labor_inc Reg_log_hours using reg_results_spec1__Q`qq'.txt, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			label title("Regression Results with Year Fixed Effects") ///
			keep(2.mh_Q`qq' 3.mh_Q`qq' ) /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Wage)" "ln(Hours)") ///
			varwidth(20)	
}	


	*local deps "log_labor_inc log_gross_inc job_hours log_hours"
	*2.mh_Q5 3.mh_Q5 4.mh_Q5 5.mh_Q5
	*2.mh_Q`qq' 3.mh_Q`qq' 4.mh_Q`qq' 5.mh_Q`qq' 6.mh_Q`qq' 7.mh_Q`qq' 8.mh_Q`qq' 9.mh_Q`qq' 10.mh_Q`qq'
	/*	
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
