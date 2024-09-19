/*

	
		
		histogram `avg_var', bin(30) ///
			title("Distribution of `avg_var'") ///
			xtitle("`avg_var'") ytitle("Frequency")
		graph export "`savedir'/`avg_var'_dist.png", replace	// Drop the temporary variable bin_cross_var if not needed
	
	
	/*
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
			*/
	/*
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
		*/
	
		// Cluster-robust standard errors by age
		gen se_`avg_var' = .
		by `cross_var': replace se_`avg_var' = sd_`avg_var' / sqrt(_N)
		gen ci_h = avg_`avg_var' + se_`avg_var'
		gen ci_l = avg_`avg_var' - se_`avg_var'

		/*
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
		*/
