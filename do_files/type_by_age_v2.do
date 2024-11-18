capture drop _clus*
capture drop MH_age_dec
egen MH_age_dec = xtile(mental_health), by(age) nq(10)

capture drop MH_clust_50p_age_flag
gen MH_clust_50p_age_flag = 0
replace MH_clust_50p_age_flag = 1 if MH_age_dec > 5

capture drop MH_50p_mean
preserve
collapse (mean) MH_clust_50p_age_flag, by(indiv_id)
rename MH_clust_50p_age_flag MH_50p_mean
tempfile clust_flag_mean
save `clust_flag_mean'
restore

// merge the mean with the data 
merge m:1 indiv_id using `clust_flag_mean'
tab _merge
drop _merge

// Greater than >50% of the time cluster/cutoff
capture drop MH_clust_50p_age
gen MH_clust_50p_age = 0
replace MH_clust_50p_age = 1 if MH_50p_mean > 0.5 

//K Means cluster/cutoff
preserve
collapse (mean) MH_50p_mean, by (indiv_id)
cluster kmeans MH_50p_mean, k(2)
rename _clus_1 MH_clust_k2_age

sum MH_50p_mean if MH_clust_k2_age == 1
local mean_MH_1 = r(mean)
di `mean_MH_1'
sum MH_50p_mean if MH_clust_k2_age == 2
local mean_MH_2 = r(mean)
di `mean_MH_2'
if `mean_MH_1' >= `mean_MH_2'{
	replace MH_clust_k2_age = 0 if MH_clust_k2_age == 2
} 
else {
	replace MH_clust_k2_age = 0 if MH_clust_k2_age == 1
	replace MH_clust_k2_age = 1 if MH_clust_k2_age == 2
}
//save 
tempfile k2_clust_data
save `k2_clust_data'
restore

// merge 
merge m:1 indiv_id using `k2_clust_data'
tab _merge
drop _merge

list indiv_id age MH_age_dec MH_50p_mean MH_clust_k2_age in 1/100

sum MH_50p_mean if MH_clust_k2_age == 1
sum MH_50p_mean if MH_clust_50p_age == 1
