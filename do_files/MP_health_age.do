cls
clear
local dir = "$datadir"

cd "`dir'"
// use "`dir'/UKHLS_sample_trans", clear
use "`dir'/UKHLS_sample_trans", clear

preserve
collapse (mean) mental_health, by(age) 
rename mental_health mean_mental_health
tempfile mean_mental_health
save `mean_mental_health', replace
restore

preserve
collapse (sd) mental_health, by(age)
rename mental_health sd_mental_health
tempfile sd_mental_health
save `sd_mental_health', replace
restore

preserve
collapse (sem) mental_health, by(age)
rename mental_health sem_mental_health
tempfile sem_mental_health
save `sem_mental_health', replace
restore

preserve
collapse (mean) physical_health, by(age)
rename physical_health mean_physical_health
tempfile mean_physical_health
save `mean_physical_health', replace
restore

preserve
collapse (sd) physical_health, by(age)
rename physical_health sd_physical_health
tempfile sd_physical_health
save `sd_physical_health', replace
restore

preserve
collapse (sem) physical_health, by(age)
rename physical_health sem_physical_health
tempfile sem_physical_health
save `sem_physical_health', replace
restore

cd "$outdir"
preserve
use `mean_mental_health', clear
merge 1:1 age using `sd_mental_health', nogen
merge 1:1 age using `sem_mental_health', nogen
merge 1:1 age using `mean_physical_health', nogen
merge 1:1 age using `sd_physical_health', nogen
merge 1:1 age using `sem_physical_health', nogen
export delimited using "MP_health_age.csv", replace
restore