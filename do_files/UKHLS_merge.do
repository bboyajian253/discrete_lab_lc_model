
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
local mergeName "UKHLS_merged.dta"
save "`mergeName'", replace

di " ***** UKHLS_merged.dta saved *****"