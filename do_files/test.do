
// capture gen log_physical_health = log(physical_health)
// capture gen log_mental_health = log(mental_health)
//
// preserve
// collapse (mean) mental_health physical_health log_mental_health log_physical_health, by(age)
// list
// restore
//
// preserve
// collapse (sd) mental_health physical_health log_mental_health log_physical_health, by(age)
// list
// restore
//

// drop if wage_percentile > 990 & emp == 1

preserve 
collapse (mean) mean_wage=log_wage (sd) sd_wage=log_wage (count) num_obs=log_wage (min) min_wage=log_wage (max) max_wage=log_wage (min) min_hours=job_hours (max) max_hours=job_hours if emp == 1, by(age)
// collapse (mean) mean_wage=wage (sd) sd_wage=wage (count) num_obs=wage (min) min_wage=wage (max) max_wage=wage (min) min_hours=job_hours (max) max_hours=job_hours if emp == 1, by(age)
list
restore

preserve 
// collapse (mean) mean_wage=log_wage (sd) sd_wage=log_wage (count) num_obs=log_wage (min) min_wage=log_wage (max) max_wage=log_wage (min) min_hours=job_hours (max) max_hours=job_hours if emp == 1, by(age)
collapse (mean) mean_wage=wage (sd) sd_wage=wage (count) num_obs=wage (min) min_wage=wage (max) max_wage=wage (min) min_hours=job_hours (max) max_hours=job_hours if emp == 1, by(age)
list
restore

sum emp job_hours wage labor_incom 
// list job_hours wage labor_incom if wage !=. & job_hours == . & emp == 1
// local ages "70 71 72 73 74 75"
// foreach a in `ages'{
// 	sum MH_G2P MH_P2G if age == `a'
// }
