
capture drop dups_id_age has_dups_id_age

sort indiv_id age
list indiv_id age wave year labor_income wage emp job_hours in 1/25

duplicates report indiv_id age
duplicates tag indiv_id age, gen(dups_id_age)
sum dups_id_age
count if dups_id_age == 1
by indiv_id: egen has_dups_id_age = max(dups_id_age)
list indiv_id age wave year month day if dups_id_age
list indiv_id age wave year month day if has_dups_id_age

// list indiv_id age wave year month day wage emp job_hours if dups_id_age == 1 in 1/1000 

duplicates report indiv_id age
duplicates report indiv_id year
duplicates report indiv_id year age

gen derived_age = year - birthyear
gen derived_year = birthyear + age

count if derived_age != age
count if derived_year != year

preserve
// DROP AFTER DUPLICATE BY indiv_id and age appears 
* First, sort the data by indiv_id and age
sort indiv_id age

* Now flag the first duplicate occurrence of age within each indiv_id
by indiv_id (age): gen byte first_dup_age = (age == age[_n-1] & indiv_id == indiv_id[_n-1])

* Create a cumulative sum to identify all observations after the first duplicate
by indiv_id: gen byte drop_obs = sum(first_dup_age)

* Keep only observations before and including the first duplicate age
keep if drop_obs == 0

duplicates report indiv_id age
duplicates report indiv_id year
duplicates report indiv_id year age

restore

