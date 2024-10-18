
cls
clear
*set trace on

local dir = "$datadir"
local savedir "$dodir/UKHLS_quants_output"

cd  "`dir'"	


// use `dir'/UKHLS_sample_clean, clear
// use UKHLS_sample, clear
use UKHLS_sample_trans, clear

rename log_labor_earnings log_lab_earn

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
// local quants = "2 3 4 5 8 10"
// local quants = "2 3 4 5"
local quants = "5"
foreach q in `quants'{
	sort physical_health
	gen physical_health_nonmissing = physical_health if physical_health != .
	capture drop ph_Q`q'
	xtile ph_Q`q' = physical_health_nonmissing [pweight=wght], nq(`q')
	drop physical_health_nonmissing  

	sort mental_health
	gen mental_health_nonmissing = mental_health if mental_health != .
	capture drop mh_Q`q'
	xtile mh_Q`q' = mental_health_nonmissing [pweight=wght], nq(`q')
	drop mental_health_nonmissing  // Sort the data in ascending order
}	

* Label the values of mh_Q5
label define mh_q5lab 1 "Poor Mental Health" 2 "Fair Mental Health" 3 "Good Mental Health" 4 "Very Good MH" 5 "Excellent Mental Health"
label values mh_Q5 mh_q5lab
label define ph_q5lab 1 "Poor Physical Health" 2 "Fair Physical Health" 3 "Good Physical Health" 4 "Very Good PH" 5 "Excellent Physical Health"
label values ph_Q5 ph_q5lab
// * Label the values of mh_Q4
// label define mh_q4lab 1 "poor mh" 2 "fair mh" 3 "good mh" 4 "very good mh"
// label values mh_Q4 mh_q4lab
// label define ph_q4lab 1 "poor ph" 2 "fair ph" 3 "good ph" 4 "very good ph" 
// label values ph_Q4 ph_q4lab
// * Label the values of mh_Q3
// label define mh_q3lab 1 "poor mh" 2 "fair mh" 3 "good mh" 
// label values mh_Q3 mh_q3lab
// label define ph_q3lab 1 "poor ph" 2 "fair ph" 3 "good ph" 
// label values ph_Q3 ph_q3lab
// * Label the values of mh_Q2
// label define mh_q2lab 1 "poor mh" 2 "fair mh" 
// label values mh_Q2 mh_q2lab
// label define ph_q2lab 1 "poor ph" 2 "fair ph" 
// label values ph_Q2 ph_q2lab
// *svyset psu [pweight=long_weights], strata(strata)
	

cd "`savedir'"
// regress deps on MH and PH 
local deps "log_lab_earn log_wage log_hours"

local FE1 "year"
local FE2 "year indiv_id"
local myFEs "FE1 FE2"

// label variable mental_health "Mental Health"
label variable MH "MH State"
label variable PH "PH State"

local specEdMH "i.MH i.PH i.MH#i.PH i.urban i.race age age2 age3 i.mar_stat i.sex i.educ [pweight=wght]"
// can add more specs here later
local spec "`specEdMH'"
local spec_name "specEdMH"
local myKeepMH  "1.MH 1.PH 2.sex 1.educ"

foreach FE in `myFEs'{

	// Run reghdfe and store the results
	reghdfe emp `spec', absorb(``FE'') 

	if ("``FE''" == "year"){
		estimates store Reg_emp_year
	} 
	else if ("``FE''" == "year indiv_id"){
		estimates store Reg_emp_year_id
	}

	foreach dep in `deps'{	
		// Run regression with year fixed effects
		reghdfe `dep' `spec' if emp == 1, absorb(``FE'')
		// Save the regression results
		if "``FE''" == "year"{
			estimates store Reg_`dep'_year
		} 
		else if "``FE''" == "year indiv_id"{
			estimates store Reg_`dep'_year_id
		}
	}
}

di "***** ran MH and PH regressions *****"


esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year ///
Reg_emp_year_id  Reg_log_lab_earn_year_id Reg_log_wage_year_id Reg_log_hours_year_id ///
using reg_results_`spec_name'_both.tex, booktabs replace cells(b(star fmt(3)) se(par fmt(3))) ///
	stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
	label /// width(\textwidth) 
	prehead(`"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"' ///
			`"\begin{table}"' ///
			`"\center\caption{Mental and Physical Health State Effects and Labor Outcomes}"' ///
			`"\footnotesize"' ///
			`"\begin{tabular}{l*{@M}{c}}"' ) ///
	postfoot(`"\hline"' ///
			`"\multicolumn{9}{l}{\small{All models control for race, marital status, urban location, and an age cubic.}} \\"' ///
			`"\end{tabular}"' ///
			`"\end{table}"' ) ///
	substitute(\midrule \hline) ///
	keep(`myKeepMH') /// Exclude all coeffecients excepet quintiles
	mgroups("Year FE" "Individ and Year FE", pattern(1 0 0 0 1 0 0 0) prefix(\multicolumn{@span}{c}{\underline{) suffix(}}) span erepeat(\cmidrule(lr){@span})) ///
	mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)" "Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)") nonumbers collabels(none)  

di "***** saved the first table *****"

***Combine and export the stored results to a LaTeX .tex file***
esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year ///
using reg_results_`spec_name'.tex, booktabs replace cells(b(star fmt(3)) se(par fmt(3))) ///
	stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
	label /// width(\textwidth) 
	prehead( `"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"' ///
			`"\begin{table}"' ///
			`"\center\caption{Mental and Physical Health State Effects and Labor Outcomes}"' ///
			`"\footnotesize"' ///
			`"\begin{tabular}{l*{@M}{c}}"' ) /// varwidth(20) width(\textwidth) 
	postfoot(	`"\hline"' ///
				`"\multicolumn{5}{l}{\small{All models control for race, marital status, urban location, an age cubic, and year.}} \\"' ///
				`"\end{tabular}"' ///
				`"\end{table}"' ) ///
	keep(`myKeepMH') /// Exclude all coeffecients excepet quintiles
	mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)") nonumbers collabels(none)  

di "***** saved the second table *****"

***QUANTS*****************	
****make the number of quants smaller for debugging purposes.
// local quants = "5 10"
local quants = "5"
****make the number of quants smaller for debugging purposes.
***QUANTS*****************		

foreach q in `quants'{	
	local qq = `q'  //what mh quantile spec are we working with?
	local pq = `q'  //what ph quant spec for now keep the same as MH quant
	*build strings for variables to keep for regressions
	local myKeepMP`q' ""
	local myKeepMH`q' ""
	local myKeepPH`q' ""
	local i = 2
	while `i'<=`q'{
		local myKeepMP`q' "`myKeepMP`q'' `i'.mh_Q`qq' `i'.ph_Q`pq' 2.sex 1.educ" 
		// local myKeepMP`q' "`myKeepMP`q'' `i'.mh_Q`qq' `i'.ph_Q`pq' 2.sex " //1.educ
		local myKeepMH`q' "`myKeepMH`q'' `i'.mh_Q`qq'"
		local myKeepPH`q' "`myKeepPH`q'' `i'.mh_Q`pq'"
		local ++i
	} 
	//end while

	local specEd "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' i.urban i.race age age2 age3 i.mar_stat i.sex i.educ [pweight=wght]"
	local specEdLvl "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' i.urban i.race age age2 age3 i.mar_stat i.sex i.educ_level [pweight=wght]"
	local specHighQual "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' i.urban i.race age age2 age3 i.mar_stat i.sex i.high_qual [pweight=wght]"
	// local specs "specEd specEdLvl specHighQual"
	local specs "specEd"
	
	local FE1 "year"
	local FE2 "year indiv_id"
	local myFEs "FE1 FE2"
	
	foreach spec in `specs'{
		foreach myFE in `myFEs'{
			di "This is the current fixed effect: "
			di "``myFE''"	
			* Run reg and store the results
			reghdfe emp ``spec'', absorb(``myFE'') 
			if ("``myFE''" == "year"){
				estimates store Reg_emp_year
			} 
			else if ("``myFE''" == "year indiv_id"){
				estimates store Reg_emp_year_id
			}
			foreach dep in `deps'{
				// Run regression with year fixed effects, using mh_q1 as the excluded group (comparison group)
				*xtreg `dep' ``spec'' if emp == 1  // `weights'
				reghdfe `dep' ``spec'' if emp == 1, absorb(``myFE'')
				// Save the regression results
				if "``myFE''" == "year"{
					estimates store Reg_`dep'_year
				} 
				else if "``myFE''" == "year indiv_id"{
					estimates store Reg_`dep'_year_id
				}
				***ENDS DEP LOOP***
			}		
		***ENDS FE LOOP***
		}
		***Combine and export the stored results to a text file***
		*CHANGE TO THE SAVE LOCATION
		cd "`savedir'"
		
		*if `qq' >= 5{
			*2.mh_Q`qq' 3.mh_Q`qq' 4.mh_Q`qq' 5.mh_Q`qq' 2.ph_Q`pq' 3.ph_Q`pq' 4.ph_Q`pq' 5.ph_Q`pq'

			// ***Combine and export the stored results to a text file***
			// esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year ///
			// Reg_emp_year_id Reg_log_lab_earn_year_id Reg_log_wage_year_id Reg_log_hours_year_id ///
			// using reg_results_Q`q'_`spec'.txt, replace cells(b(star fmt(3)) se(par fmt(3))) ///
			// 	stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
			// 	label title("Regression Results with Year Fixed Effects") ///
			// 	keep(`myKeepMP`q'') /// Exclude all coeffecients excepet quintiles
			// 	mgroups("Year FE" "Indiv & Year", pattern(1 0 0 0 1 0 0 0)) ///
			// 	mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)"  "Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)")  ///
			// 	varwidth(20)
			// 	*title("Regression Results with Interacations and Year Fixed Effects") 

			// ***Combine and export the stored results to a LaTeX .tex file***
			
		esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year ///
		Reg_emp_year_id  Reg_log_lab_earn_year_id Reg_log_wage_year_id Reg_log_hours_year_id ///
		using reg_results_Q`q'_`spec'_both.tex, booktabs replace cells(b(star fmt(3)) se(par fmt(3))) ///
			stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
			label /// width(\textwidth) 
			prehead(`"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"' ///
					`"\begin{table}"' ///
					`"\center\caption{Mental and Physical Health Quintile Effects and Labor Outcomes}"' ///
					`"\footnotesize"' ///
					`"\begin{tabular}{l*{@M}{c}}"' ) ///
			postfoot(`"\hline"' ///
					`"\multicolumn{9}{l}{\small{All models control for race, marital status, urban location, and an age cubic.}} \\"' ///
					`"\end{tabular}"' ///
					`"\end{table}"' ) ///
			substitute(\midrule \hline) ///
			keep(`myKeepMP`q'') /// Exclude all coeffecients excepet quintiles
			mgroups("Year FE" "Individ and Year FE", pattern(1 0 0 0 1 0 0 0) prefix(\multicolumn{@span}{c}{\underline{) suffix(}}) span erepeat(\cmidrule(lr){@span})) ///
			mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)" "Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)") nonumbers collabels(none)  
			
		***Combine and export the stored results to a LaTeX .tex file***
		esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year ///
		using reg_results_Q`q'_`spec'.tex, booktabs replace cells(b(star fmt(3)) se(par fmt(3))) ///
			stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
			label /// width(\textwidth) 
			prehead( `"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"' ///
					`"\begin{table}"' ///
					`"\center\caption{Mental and Physical Health Quintile Effects and Labor Outcomes}"' ///
					`"\footnotesize"' ///
					`"\begin{tabular}{l*{@M}{c}}"' ) /// varwidth(20) width(\textwidth) 
			postfoot(	`"\hline"' ///
						`"\multicolumn{5}{l}{\small{All models control for race, marital status, urban location, an age cubic, and year.}} \\"' ///
						`"\end{tabular}"' ///
						`"\end{table}"' ) ///
			keep(`myKeepMP`q'') /// Exclude all coeffecients excepet quintiles
			mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)") nonumbers collabels(none)  
				// substitute(\midrule \hline) ///
	} 
	***end of specs****
}	
		****QUANTS ENDS****
	
********************************
***Continuous measures stuff****
********************************
label variable mental_health "Mental Health"
label variable physical_health "Physical Health"
gen MHxPH = physical_health*mental_health
label variable MHxPH "MH $\times$ PH"

local FE1 "year"
local FE2 "year indiv_id"
local myFEs "FE1 FE2"

foreach myFE in `myFEs'{	

	*c.mental_health#c.physical_health
	// local specCont "mental_health physical_health MHxPH i.urban i.race age age2 age3 i.mar_stat i.sex i.educ [pweight=wght]"
	local specCont "mental_health physical_health MHxPH i.urban i.race age age2 age3 i.mar_stat i.sex i.educ [pweight=wght]"
	local myKeepCont "mental_health physical_health MHxPH 2.sex 1.educ"
	*eventually I can put a loop here if needed
	local spec specCont
	
	* Run xtreg and store the results
	reghdfe emp ``spec'', absorb(``myFE'') //`weights'
	*estimates store Reg_emp
	if ("``myFE''" == "year"){
		estimates store Reg_emp_year
	}  
	else if ("``myFE''" == "year indiv_id"){
		estimates store Reg_emp_year_id
	}
	
	foreach dep in `deps'{

		// Run regression with year fixed effects, using mh_q1 as the excluded group (comparison group)
		
		*xtreg `dep' ``spec'' if emp == 1  // `weights'
		reghdfe `dep' ``spec'' if emp == 1, absorb(``myFE'')
		// Save the regression results
		if "``myFE''" == "year"{
			estimates store Reg_`dep'_year
		} 
		else if "``myFE''" == "year indiv_id"{
			estimates store Reg_`dep'_year_id
		}
	}
	
}

//end fe loop
***for table format testing
esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year ///
	Reg_emp_year_id Reg_log_lab_earn_year_id Reg_log_wage_year_id Reg_log_hours_year_id ///
	using reg_results_`spec'_both.tex, booktabs replace cells(b(star fmt(5)) se(par fmt(3))) ///
	stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
	label /// width(\textwidth)
	prehead( `"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"' ///
	`"\center\caption{Mental and Physical Health Index Effects and Labor Outcomes}"' ///
	`"\tiny\begin{tabular}{l*{@M}{c}}"' ) /// varwidth(20) width(\textwidth)
	substitute(\midrule \hline) ///
	postfoot(`"\tabnotes{9}{All models also control for sex, a cubic in age, race, marital status, urban location and year.}"' ) ///
	keep(`myKeepCont') /// Exclude all coeffecients excepet quintiles
	mgroups("Year FE" "Individ and Year FE", pattern(1 0 0 0 1 0 0 0) prefix(\multicolumn{@span}{c}{\underline{) suffix(}}) span erepeat(\cmidrule(lr){@span})) ///
		mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)" "Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)") nonumbers collabels(none) //
	*title("Mental and Physical Health Effects on Labor Outcomes")
	*`"\tiny"' ///
	*span erepeat(\cmidrule(lr){@span}
	
esttab Reg_emp_year Reg_log_lab_earn_year Reg_log_wage_year Reg_log_hours_year using reg_results_`spec'.tex, booktabs replace cells(b(star fmt(5)) se(par fmt(3))) ///
	stats(N r2 r2_a, labels("Observations" "R-Square" "Adj. R-Square")) ///
	label /// width(\textwidth)
	prehead( `"\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"' ///
	`"\center\caption{Mental and Physical Health Index Effects and Labor Outcomes}"' ///
	`"\tiny\begin{tabular}{l*{@M}{c}}"' ) /// varwidth(20) width(\textwidth)
	substitute(\midrule \hline) ///
	postfoot(`"\tabnotes{5}{All models also control for sex, a cubic in age, race, marital status, urban location and year.}"') ///
	keep(`myKeepCont') /// Exclude all coeffecients excepet quintiles
	mtitles("Employment" "ln(Earnings)" "ln(Wage)" "ln(Hours)") nonumbers collabels(none) //
	*title("Mental and Physical Health Effects on Labor Outcomes")
	*`"\tiny"' ///
	*span erepeat(\cmidrule(lr){@span}

cd "`dir'"

rename log_lab_earn log_labor_earnings

save UKHLS_sample_quants, replace

 di "*****got to the end of UKHLS_quants.do*****"