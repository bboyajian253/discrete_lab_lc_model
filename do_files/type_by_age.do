// typed by age
// preserve
capture drop MH_age_dec
egen MH_age_dec = xtile(mental_health), by(age) nq(10)
// local age_list "25 35 45 55 65 75"
local age_list "25 45 65"
levelsof MH_age_dec, local(dec_levels)
foreach age of local age_list {
    foreach dec of local dec_levels {
        sum mental_health if MH_age_dec == `dec' & age == `age'
    }
}

// xtset indiv_id age
// tab age MH_age_dec
capture drop MH_clust_50p_age_flag
gen MH_clust_50p_age_flag = 0
replace MH_clust_50p_age_flag = 1 if MH_age_dec >= 6

// store the mean of the flag
capture drop MH_clust_50p_age_mean
preserve
collapse (mean) MH_clust_50p_age_flag, by(indiv_id)
rename MH_clust_50p_age_flag MH_clust_50p_age_mean
tempfile clust_flag_mean
save `clust_flag_mean'
restore

// merge the mean back to the data
merge m:1 indiv_id using `clust_flag_mean'
tab _merge
drop _merge

sort indiv_id age
list indiv_id age MH_age_dec MH_clust_50p_age_flag MH_clust_50p_age_mean in 1/100

capture drop MH_clust_50p_age
gen MH_clust_50p_age = 0
replace MH_clust_50p_age = 1 if MH_clust_50p_age_mean <= 0.5 

// restore