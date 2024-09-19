
	cls
	clear
    pause on
	
	
	local dir = "$datadir"
	local savedir "$outdir/UKHLS_quant"
	
	cd  `dir'	
	
	
	use UKHLS_sample, clear
	***drop non men***
	*keep if sex==1
	
	capture drop mh_q*
	
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
	label define mh_q5lab 1 "Poor Mental Health" 2 "Fair Mental Health" 3 "Good Mental Health" 4 "Very Good MH" 5 "Excellent Mental Health"
	label values mh_Q5 mh_q5lab
	label define ph_q5lab 1 "Poor Physical Health" 2 "Fair Physical Health" 3 "Good Physical Health" 4 "Very Good PH" 5 "Excellent Physical Health"
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
	
	****make the number of quants smaller for debugging purposes.
	local quants = "5"
	****make the number of quants smaller for debugging purposes.
	
	local deps "log_labor_inc log_hours"
	local spec1 "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' urban i.race age age2 age3 i.mar_stat sex"
	local specs "spec1"
	*fpreach spec in specs (then replace all the spec1s below iwth some form of `spec'
	
cd "`savedir'"
foreach q in `quants'{
	
	local qq = `q' //what mh quantile are we working with?
	local pq = `q'  //what ph quant for now keep the same as MH quant
	
	local spec1 "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' urban i.race age age2 age3 i.mar_stat sex"
	local spec2 "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' urban i.race age age2 age3 i.mar_stat sex i.educ"
	****************************************************************
	****CAN CHANGE THE BELOW LINE TO RUN THROUGH DIFFERENT SPECS****
	****be careful with storage of regression results***************
	****************************************************************
	local specs "spec1 spec2"
	foreach spec in `specs'{
	
	* Run xtreg and store the results
	xtreg emp ``spec'' //`weights'
	estimates store Reg_emp
	*pause
	* Perform multiple coeff equality tests and store the results
	test 2.mh_Q`qq'==0
	mat m0 =(r(chi2), r(p))
	mat rowname m0="Test 0"

	
	local m m0
	
	mata : st_local("all_locals", invtokens(st_dir("local", "macro", "*")'))
	display "ALL LOCALS: `all_locals'"
	macro list

	if `qq' > 2 {
		test 2.mh_Q`qq'=3.mh_Q`qq'
		mat m1 =(r(chi2), r(p))
		mat rowname m1="Test 1"
		
		local m `m' \m1
		
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
		
	}
	mat m = `m'
	* Export the results to an Excel file using putexcel
	
	* Export the results to an Excel file using putexcel
	putexcel set "coeff_tests_Q`qq'.xlsx", sheet(emp) modify
	putexcel B1= "Chi2"
	putexcel C1= "P Value"
	putexcel A2 = "2.mh_Q`qq' == 0"
	putexcel A3= "2.mh_Q`qq' - 3.mh_Q`qq' = 0"
	putexcel A4= "3.mh_Q`qq' - 4.mh_Q`qq' = 0"
	putexcel A5= "4.mh_Q`qq' - 5.mh_Q`qq' = 0"
	
	di "GOT HERE!!!!!!!!!!!!!!!!!!!"
		mata : st_local("all_locals", invtokens(st_dir("local", "macro", "*")'))
		display "ALL LOCALS: `all_locals'"
		macro list
	
	putexcel B2= matrix(m)


	foreach dep in `deps'{

		// Run regression with year fixed effects, using mh_q1 as the excluded group (comparison group)
		
		****************************************************************
		****CAN CHANGE THE BELOW LINE TO RUN THROUGH DIFFERENT SPECS****
		****be careful with storage of regression results***************
		****************************************************************
		
		xtreg `dep' ``spec'' if emp == 1  // `weights'
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
		local m `m' \m1
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
		putexcel set "coeff_tests_Q`qq'_`spec'.xlsx", sheet(`dep') modify
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
	cd "`savedir'"
	

	if `qq' >= 5{

		esttab Reg_emp Reg_log_labor_inc Reg_log_hours using reg_results_Q`qq'_`spec'.txt, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			label title("Regression Results with Year Fixed Effects") ///
			keep(2.mh_Q`qq' 3.mh_Q`qq' 4.mh_Q`qq' 5.mh_Q`qq' 2.ph_Q`pq' 3.ph_Q`pq' 4.ph_Q`pq' 5.ph_Q`pq') /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Wage)" "ln(Hours)") ///
			varwidth(20)
			*title("Regression Results with Interacations and Year Fixed Effects") 
		esttab Reg_emp Reg_log_labor_inc Reg_log_hours using reg_results_Q`qq'_`spec'.tex, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			label width(\textwidth) ///
			keep(2.mh_Q`qq' 3.mh_Q`qq' 4.mh_Q`qq' 5.mh_Q`qq' 2.ph_Q`pq' 3.ph_Q`pq' 4.ph_Q`pq' 5.ph_Q`pq') /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Wage)" "ln(Hours)") nonumbers collabels(none) ///
			varwidth(20)
	}
	else if `qq' >= 4{
			
		esttab Reg_emp Reg_log_labor_inc Reg_log_hours using reg_results_Q`qq'_`spec'.txt, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			label title("Regression Results with Year Fixed Effects") ///
			keep(2.mh_Q`qq' 3.mh_Q`qq' 4.mh_Q`qq') /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Wage)" "ln(Hours)") ///
			varwidth(20)
	}
	else if `qq' >= 3{

		esttab Reg_emp Reg_log_labor_inc Reg_log_hours using reg_results_Q`qq'_`spec'.txt, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			label title("Regression Results with Year Fixed Effects") ///
			keep(2.mh_Q`qq' 3.mh_Q`qq') /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Wage)" "ln(Hours)") ///
			varwidth(20)
	} 
	else if `qq' >= 2{

		esttab Reg_emp Reg_log_labor_inc Reg_log_hours using reg_results_Q`qq'_`spec'.txt, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			label title("Regression Results with Year Fixed Effects") ///
			keep(2.mh_Q`qq' 2.ph_Q`pq') /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Wage)" "ln(Hours)") ///
			varwidth(20)
	} 
	}
}	

cd `dir'

save UKHLS_sample_quants, replace
	
	

	
	
