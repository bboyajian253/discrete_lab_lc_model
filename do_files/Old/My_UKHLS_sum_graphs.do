local saveLoc "C:\Users\Ben\My Drive\PhD\PhD Year 3\3rd Year Paper\Data\Stata Clean Etc\Regression Output"

local avg_var mental_health
local cross_var age
local cross_var2 year

***PRESERVE****
preserve 

***dos osme cleaning***
drop if age > 70

***gen avg avg_var by cross_var
capture egen avg_`avg_var' = mean(`avg_var'), by(`cross_var')
***sort by cross_var
sort `cross_var'
***plot avg avg_var by cross_var
twoway (line avg_`avg_var' `cross_var', ///
title("Average `avg_var' by `cross_var'") ///
xtitle("`cross_var'") ytitle("Average `avg_var'")) ///
(lfit avg_`avg_var' `cross_var')
graph export "`saveLoc'\testMHbyAge2009.png", replace


***gen avg avg_var by cross_var
capture egen sd_`avg_var' = sd(`avg_var'), by(`cross_var')
***sort by cross_var
sort `cross_var'
***plot avg avg_var by cross_var
twoway (line sd_`avg_var' `cross_var', ///
title("Standard deviation of `avg_var' by `cross_var'") ///
xtitle("`cross_var'") ytitle("SD `avg_var'")) ///
(lfit sd_`avg_var' `cross_var')
graph export "`saveLoc'\testMHSDbyAge2009.png", replace


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
	
graph export "`saveLoc'\testMHbyAgebyYear.png", replace

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
	
graph export "`saveLoc'\testMHVarbyAgebyYear.png", replace

restore

restore
