cls
// clear

// local dir = "$datadir"
// cd  "`dir'"	
// use "`dir'/UKHLS_sample", clear

local q = 5
sort mental_health
gen mental_health_nonmissing = mental_health if mental_health != .
capture drop mh_Q`q'
xtile mh_Q`q' = mental_health_nonmissing [pweight=wght], nq(`q')
drop mental_health_nonmissing  // Sort the data in ascending order

sort physical_health
gen physical_health_nonmissing = physical_health if physical_health != .
capture drop ph_Q`q'
xtile ph_Q`q' = physical_health_nonmissing [pweight=wght], nq(`q')
drop physical_health_nonmissing  

* define current physical health 5 types
gen MH5 = mh_Q5
label define MH5lab 1 "poor" 2 "fair" 3 "good" 4 "very good" 5 "excellent"
label values MH5 MH5lab

* define current mental health 4 types
gen MH4 = mh_Q5
recode MH4 (1=1) (2=2) (3=3) (4/5=4)
label define MH4lab 1 "poor" 2 "fair" 3 "good" 4 "very good"
label values MH4 MH4lab

* define current mental health 3 types
gen MH3 = mh_Q5
recode MH3 (1=1) (2=2) (3/5=3)
label define MH3lab 1 "poor" 2 "fair" 3 "good"
label values MH3 MH3lab

* define current mental health 2 types
gen MH = mh_Q5
recode MH (1/2=0) (3/5=1) 
label define MHlab 0 "poor" 1 "good" 
label values MH MHlab

gen badMH = .
replace badMH = 0 if MH == 1
replace badMH = 1 if MH == 0

* define current physical health 5 types
gen PH5 = ph_Q5
label define PH5lab 1 "poor" 2 "fair" 3 "good" 4 "very good" 5 "excellent"
label values PH5 PH5lab

* define current physical health 4 types
gen PH4 = ph_Q5
recode PH4 (1=1) (2=2) (3=3) (4/5=4)
label define PH4lab 1 "poor" 2 "fair" 3 "good" 4 "very good"
label values PH4 PH4lab

* define current physical health 3 types
gen PH3 = ph_Q5
recode PH3 (1=1) (2=2) (3/5=3)
label define PH3lab 1 "poor" 2 "fair" 3 "good"
label values PH3 PH3lab

* define current physical health 2 types
gen PH = ph_Q5
recode PH (1/2=0) (3/5=1) 
label define PHlab 0 "poor" 1 "good" 
label values PH PHlab

gen badPH = .
replace badPH = 0 if PH == 1
replace badPH = 1 if PH == 0