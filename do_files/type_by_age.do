// typed by age
// preserve
capture drop MH_age_dec
egen MH_age_dec = xtile(mental_health), by(age) nq(10)

capture drop MH_clust_50p_age_flag
gen MH_clust_50p_age_flag = 0
replace MH_clust_50p_age_flag = 1 if MH_age_dec > 5

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
replace MH_clust_50p_age = 1 if MH_clust_50p_age_mean > 0.5 

* K Means cluster
preserve
sort MH_clust_50p_age_mean
collapse (mean) MH_clust_50p_age_mean, by(indiv_id)
cluster kmeans MH_clust_50p_age_mean, k(2)
rename _clus_1 MH_clust_k2_age
tempfile clusterdata
save `clusterdata'
restore

merge m:1 indiv_id using `clusterdata'
tab _merge
drop _merge
pause
pause on
pause

sum MH_clust_50p_age_mean if MH_clust_k2_age == 1
local mean_MH_1 = r(mean)
di `mean_MH_1'
sum MH_clust_50p_age_mean if MH_clust_k2_age == 2
local mean_MH_2 = r(mean)
di `mean_MH_2'

if `mean_MH_1' >= `mean_MH_2'{
	replace MH_clust_k2_age = 0 if MH_clust_k2_age == 2
} 
else {
	replace MH_clust_k2_age = 0 if MH_clust_k2_age == 1
	replace MH_clust_k2_age = 1 if MH_clust_k2_age == 2
}

di "*****got to the end of type_by_age.do*****"