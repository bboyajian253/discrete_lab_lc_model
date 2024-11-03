preserve
* Step 1: Create a variable to flag individuals at age 25
gen age25 = age == 25

* Step 2: Create a variable to flag individuals at age 26
gen age26 = age == 26

* Step 3: Collapse data to create two flags for each individual
collapse (max) age25 age26, by(indiv_id)

* Step 4: Calculate the total number of people at age 26
count if age26 == 1
local total_at_26 = r(N)

* Step 5: Calculate the number of people present at both age 25 and age 26
count if age25 == 1 & age26 == 1
local in_both_ages = r(N)

* Step 6: Calculate the number of new people at age 26
local new_at_26 = `total_at_26' - `in_both_ages'

* Step 7: Calculate the percentage of people present at both ages
display "Share at age 26 who were also at age 25: " (`in_both_ages' / `total_at_26') * 100 "%"

* Step 8: Calculate the percentage of new people at age 26
display "Percentage new at age 26: " (`new_at_26' / `total_at_26') * 100 "%"
restore
* Step 1: Sort the data by indiv_id and age
sort indiv_id age

* Step 2: Get all unique ages in the dataset
levelsof age, local(ages)

* Initialize the Mata matrix with correct dimensions
mata: M = J(0, 3, .)  // Empty matrix with 3 columns for Age, Share_in_both, Share_new

* Step 3: Loop through consecutive age pairs
local num_ages = wordcount("`ages'")

forval i = 1/`=`num_ages'-1' {
    local age1 = word("`ages'", `i')
    local age2 = word("`ages'", `=`i'+1')

    * Preserve dataset to restore after collapsing
    preserve

    * Step 1: Create a variable to flag individuals at age1
    gen age`age1' = age == `age1'

    * Step 2: Create a variable to flag individuals at age2
    gen age`age2' = age == `age2'

    * Step 3: Collapse data to create two flags for each individual
    collapse (max) age`age1' age`age2', by(indiv_id)

    * Step 4: Calculate the total number of people at age2
    count if age`age2' == 1
    local total_at_age2 = r(N)

    * Step 5: Calculate the number of people present at both age1 and age2
    count if age`age1' == 1 & age`age2' == 1
    local in_both_ages = r(N)

    * Step 6: Calculate the number of new people at age2
    local new_at_age2 = `total_at_age2' - `in_both_ages'

    * Step 7: Calculate the percentage of people present at both ages
    local share_in_both = (`in_both_ages' / `total_at_age2') * 100

    * Step 8: Calculate the percentage of new people at age2
    local share_new_at_age2 = (`new_at_age2' / `total_at_age2') * 100

    * Store results in the Mata matrix
    mata: M = M \ (`age2', `share_in_both', `share_new_at_age2')

    * Restore original dataset
    restore
}

* Display the Mata matrix with results
mata: M
