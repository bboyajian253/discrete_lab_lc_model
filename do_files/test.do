
// local specEd "i.mh_Q`qq' i.ph_Q`pq' i.mh_Q`qq'#i.ph_Q`pq' i.urban i.race age age2 age3 i.mar_stat i.sex i.educ [pweight=wght]"
// local spec "i.MH age age2 i.educ i.sex i.mar_stat i.race i.urban [pweight=wght]"
// local spec = "i.MH age age2 i.year i.educ"
// local spec = "i.MH age age2 i.year" // Define the controls

// Define the dependent variables
local dep_vars "emp log_labor_earnings log_wage log_hours log_hours_decimal"

// Define the controls
local controls "age age2 age3 i.sex i.mar_stat i.educ i.race i.urban i.MH_clust_50p_age"

// Tokenize the controls list to get the number of elements
local num_controls : word count `controls'

foreach dep_var in `dep_vars' {
    local curr_controls "i.year"
    // Loop through controls with an index
    forvalues i = 1/`num_controls' {
        // Extract the control variable by its position
        local control : word `i' of `controls'
        
        local curr_controls = "`curr_controls' `control'"
        if "`dep_var'" != "emp" {
            reg `dep_var' i.MH `curr_controls' [pweight=wght] if emp == 1
        }
        else {
            reg `dep_var' i.MH `curr_controls' [pweight=wght]
        }
        local MH_coef_`dep_var'_`i' = _b[1.MH]
        // di "MH == 1 coeff with `control': `MH_coeff_`control''"
    }
}

foreach dep_var in `dep_vars' {
    forvalues i = 1/`num_controls' {
        // Extract the control variable by its position
        local control : word `i' of `controls'
        di "*`dep_var'* Coeff. MH == 1 after adding `control': `MH_coef_`dep_var'_`i''"
    }
}

// reg log_wage `spec' if emp == 1

// Store the coefficient for MH == 1
// local mycoeff = _b[1.MH]
// di "Coefficient for MH == 1: `mycoeff'"

di "*****got to the end of test.do*****"

