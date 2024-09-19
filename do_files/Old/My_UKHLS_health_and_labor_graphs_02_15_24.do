	cls
	clear
     
	local dir = "$datadir"
	local savedir "$outdir/UKHLS_graphs_heatlhLab"
	
	cd  `dir'	
	
	use UKHLS_sample_probs, clear
	local avg_vars "labor_income"
	*local avg_var "labor_income"
	local cross_var age
	local cross_var2 year
	
	local cross_vars "mental_health physical_health"
	
	local minwage = 1700.00 
	
foreach cross_var in `cross_vars'{	
	foreach avg_var in `avg_vars'{

		***PRESERVE****
		preserve 
		

		***do some cleaning***
		drop if age > 55
		drop if age < 25
		drop if labor_income < `minwage'
		drop if sex != 1
		sort `cross_var'
	
		// Define the bin width for cross_var
		local bin_width = 2.5

		// Create bins for cross_var using the cut() function
		egen bin_`cross_var' = cut(`cross_var'), at(0(`bin_width')100)

		// Calculate the mean of avg_var within each bin
		egen avg_`avg_var' = mean(`avg_var'), by(bin_`cross_var')
	
		egen sd_`avg_var' = sd(`avg_var'), by(bin_`cross_var')

		***sort by cross_var
		sort `cross_var'
		***plot avg avg_var by cross_var
		*twoway (rcap ci_l ci_h `cross_var')
		twoway ///
			(line avg_`avg_var' bin_`cross_var', ///
			title("Average `avg_var' by `cross_var'") ///
			xtitle("`cross_var'") ytitle("Average `avg_var'") legend(off)) ///
		(lfit avg_`avg_var' `cross_var')
		graph export "`savedir'/`avg_var'_by`cross_var'.png", replace

		***sort by cross_var
		sort `cross_var'
		***plot avg avg_var by cross_var
		twoway (line sd_`avg_var' bin_`cross_var', ///
			title("Standard deviation of `avg_var' by `cross_var'") ///
			xtitle("`cross_var'") ytitle("SD `avg_var'")) ///
			(lfit sd_`avg_var' `cross_var')
		graph export "`savedir'/`avg_var'_SDby`cross_var'.png", replace
	
		
		restore
	}
}
