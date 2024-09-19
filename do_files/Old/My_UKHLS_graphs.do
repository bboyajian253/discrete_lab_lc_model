	cls
	clear
     
	local dir = "$datadir"
	local savedir "$outdir/UKHLS_graphs"
	
	cd  `dir'	
	
	use UKHLS_sample_probs, clear
	local avg_vars "mental_health physical_health MH PH"
	*local avg_vars "mental_health"
	local cross_var age
	local cross_var2 year
	
foreach avg_var in `avg_vars'{	

	
	
	summarize `avg_var'
	local unconditional_mean = r(mean)
	local unconditional_sd = r(sd)

	***PRESERVE****
	preserve 

	***do some cleaning***
	drop if age > 90
	
	sort `cross_var'
	***gen avg avg_var by cross_var
	bysort `cross_var': egen avg_`avg_var' = mean(`avg_var')
	egen sd_`avg_var' = sd(`avg_var'), by(`cross_var')
	// Cluster-robust standard errors by age
	gen se_`avg_var' = .
	by `cross_var': replace se_`avg_var' = sd_`avg_var' / sqrt(_N)
	gen ci_h = avg_`avg_var' + se_`avg_var'
	gen ci_l = avg_`avg_var' - se_`avg_var'

	***sort by cross_var
	sort `cross_var'
	***plot avg avg_var by cross_var
	twoway (rcap ci_l ci_h `cross_var') ///
		(line avg_`avg_var' `cross_var', ///
		title("Average `avg_var' by `cross_var'") ///
		xtitle("`cross_var'") ytitle("Average `avg_var'") legend(off)) ///
		(lfit avg_`avg_var' `cross_var')
	graph export "`savedir'/`avg_var'_by`cross_var'.png", replace

	// Calculate the z-scores based on the unconditional mean and standard deviation
	gen z_`avg_var' = (avg_`avg_var' - `unconditional_mean') / `unconditional_sd'
	
	***sort by cross_var
	sort `cross_var'
	***plot avg avg_var by cross_var
	twoway (line z_`avg_var' `cross_var', ylabel(-2 -1.5 -1 -.5 0 .5 1 1.5 2)  ///
		title("Average `avg_var' by `cross_var' scaled y-axis") ///
		xtitle("`cross_var'") ytitle("Average `avg_var'")) ///
		(lfit z_`avg_var' `cross_var')
	graph export "`savedir'/`avg_var'_by`cross_var'_scaled.png", replace

	***sort by cross_var
	sort `cross_var'
	***plot avg avg_var by cross_var
	twoway (line sd_`avg_var' `cross_var', ///
		title("Standard deviation of `avg_var' by `cross_var'") ///
		xtitle("`cross_var'") ytitle("SD `avg_var'")) ///
		(lfit sd_`avg_var' `cross_var')
	graph export "`savedir'/`avg_var'_SDby`cross_var'.png", replace


	drop avg_`avg_var'
	egen avg_`avg_var' = mean(`avg_var'), by(`cross_var' `cross_var2')		
	sort `cross_var' `cross_var2'
	twoway (line avg_`avg_var' `cross_var' if `cross_var2' == 2010,  ///
		title("Average `avg_var' by `cross_var'") ///
		xtitle("`cross_var'") ytitle("Average `avg_var'") ///
		legend(label(1 "Year 2010"))) /// 
		(line avg_`avg_var' `cross_var' if `cross_var2' == 2015, ///
		legend(label(2 "Year 2015"))) ///
		(line avg_`avg_var' `cross_var' if `cross_var2' == 2020, ///
		legend(label(3 "Year 2020"))) ///
		(line avg_`avg_var' `cross_var' if `cross_var2' == 2021, ///
		legend(label(4 "Year 2021"))) ///
		(lfit avg_`avg_var' `cross_var')
	
	graph export "`savedir'/`avg_var'_by`cross_var'byYear.png", replace

	drop sd_`avg_var'
	egen sd_`avg_var' = sd(`avg_var'), by(`cross_var' `cross_var2')		
	sort `cross_var' `cross_var2'
	twoway (line sd_`avg_var' `cross_var' if `cross_var2' == 2010,  ///
		title("SD `avg_var' by `cross_var'") ///
		xtitle("`cross_var'") ytitle("Standard deviation `avg_var'") ///
		legend(label(1 "Year 2010"))) /// 
		(line sd_`avg_var' `cross_var' if `cross_var2' == 2015, ///
		legend(label(2 "Year 2015"))) ///
		(line sd_`avg_var' `cross_var' if `cross_var2' == 2020, ///
		legend(label(3 "Year 2020"))) ///
		(line sd_`avg_var' `cross_var' if `cross_var2' == 2021, ///
		legend(label(4 "Year 2021"))) ///
		(lfit sd_`avg_var' `cross_var')
	
	graph export "`savedir'/`avg_var'_sdby`cross_var'byYear.png", replace
	
	
	histogram `avg_var', bin(30) ///
		title("Distribution of `avg_var'") ///
		xtitle("`avg_var'") ytitle("Frequency")
	graph export "`savedir'/`avg_var'_dist.png", replace
	
	restore
}

/*

	cls
	clear
     
	local dir = "$datadir"
	local savedir "$outdir/UKHLS_graphs"
	
	cd  `dir'	
	
	use UKHLS_sample_probs, clear	
	local avg_var  mental_health
	local cross_var `cross_var'
	capture drop ci_*
	// Calculate mean and standard deviation
	summarize `avg_var'
	local unconditional_mean = r(mean)
	local unconditional_sd = r(sd)
	sort `cross_var'
	***gen avg avg_var by cross_var
	quietly egen avg_`avg_var' = mean(`avg_var'), by(`cross_var')
	quietly egen sd_`avg_var' = sd(`avg_var'), by(`cross_var')
	// Cluster-robust standard errors by age
	by age: replace sd_`avg_var' = sd_`avg_var' / sqrt(_N)
	rename sd_`avg_var' se_`avg_var'
	gen ci_h = avg_`avg_var' + se_`avg_var'
	gen ci_l = avg_`avg_var' - se_`avg_var'
	
	list ci*
	
	***sort by cross_var
	sort `cross_var'
	***plot avg avg_var by cross_var
	twoway (rcap ci_l ci_h `cross_var') ///
		(line avg_`avg_var' `cross_var', ///
        title("Average `avg_var' by `cross_var' with Clustered Standard Errors") ///
		xtitle("`cross_var'") ytitle("Average `avg_var'")) ///
		(lfit avg_`avg_var' `cross_var')
		
	//export graph	
	graph export "`savedir'/`avg_var'_byAge_test_with_se.png", replace
	
	
