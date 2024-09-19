***BASED ON CODE WITH THE FOLLOWING HEADING***
* Estimate health shock process. 
*    updated: Dec 1, 2023

* Required input data (saved in 'curdir')
*    - HRS_PRsurvival.dta (output file from HRS_estSurvivalProb.do)
*    - PSID_healthshock.dta (output file automatically created from PSID_documentation1.do)
*
* Output files (saved in 'outdir')er
*    - Table2_Table2.tex containing estimated of health shock process (Table 1 and Table 2)


// global curdir = "/Users/p-porapakkarm/Library/CloudStorage/Dropbox/HealthFX/ReStud_Replication/PSID"
// global outdir = "$curdir/PSIDout"

cd  "$datadir"


cap log close

****************************************
*** Estimate health process using ML ***
****************************************
cls 
clear

**** define log-likelihood function ****
capture program drop _all
program define healthML_d0  
	args     todo b lnf
	tempvar  xb1 xb2 xb3  PPp1 PPp2 PPp3  PFp1 PFp2 PFp3  PGp1 PGp2 PGp3  Peta1 Peta2 Peta3  Lf_1t  Lf_2t  Lf_3t  Lf_1 Lf_2  Lf_3  na     
	tempname etaB1 etaB2  etaB3  etaH1 etaH2 etaH3  dd  ddB ddG
	
	mleval `xb1'    = `b',  eq(1)          // Pr(B(t+1)|B(t),age,age2) 
	mleval `xb2'    = `b',  eq(2)          // Pr(B(t+1)|G(t),age,age2) 
	mleval `xb3'    = `b',  eq(3)          // Pr(eta_i|ui)
	mleval `ddB'    = `b',  eq(4)  scalar
	mleval `ddG'    = `b',  eq(5)  scalar
	mleval `dd'     = `b',  eq(6)  scalar	
 	mleval `etaB1'  = `b',  eq(7)  scalar
	mleval `etaB3'  = `b',  eq(8)  scalar
 	mleval `etaH1'  = `b',  eq(9)  scalar
	mleval `etaH3'  = `b',  eq(10) scalar

	scalar `etaB1'  =  exp(`etaB1')       // etaB1> etaB2=0 > etaB1
	scalar `etaB3'  = -exp(`etaB3')	
	scalar `etaB2'  =  0
 	scalar `etaH2'  =  0
	
	scalar `ddB'    = -exp(`ddB')
	scalar `ddG'    = -exp(`ddG')
 	scalar `dd'     =  exp(`dd')   	
	
	local Hp     $ML_y1  // bad or good next period
	local poorHp $ML_y2  // poor health next period
	local Hcur   $ML_y3  // bad or good this period
    
	
	quietly {   
		forvalue i=1/3 {
			gen double `PPp`i''  = (1-`Hcur')*$ps*invlogit(`xb1'+`ddB'+`etaB`i'')  + `Hcur'*$ps*invlogit(`xb2'+`ddG'+`etaH`i'')         // Pr(P(t+1)|health(t))
			gen double `PFp`i''  = (1-`Hcur')*$ps*invlogit(`xb1'+`etaB`i'')        + `Hcur'*$ps*invlogit(`xb2'+`etaH`i'') - `PPp`i''    // Pr(F(t+1)|health(t))
			gen double `PGp`i''  =  1-`PFp`i''-`PPp`i''                                                                                 // Pr(G(t+1|health(t)))
			gen double `Lf_`i't' = (1-`Hp')*(`poorHp'*`PPp`i''+(1-`poorHp')*`PFp`i'' ) + `Hp'*`PGp`i''    // likelyhood function of being poor/fair/good in t+1
		   egen `na'= total(`Lf_`i't'==.), by ($id)                                       // count number of missing in each individual
		   egen double `Lf_`i'' = total(log(`Lf_`i't')), by( $id ) missing                // sum log-likelihood for each individual (conditioned on type i)
		   replace `Lf_`i'' = cond(`na'==0, exp(`Lf_`i''), .)                             // if there is one missing in each individual, set sum of log-likelihood in the group to missing
		   by $id : replace `Lf_`i'' = 1 if (_n!=1 & `Lf_`i''!=.)                         // set all likelihood function to 1 except the first observation
		   drop `na'
		}
		
 		by $id : gen double `Peta1' = cond(_n==1, invlogit(`xb3'), 0)                     // probability to be eta1 type 
 		by $id : gen double `Peta2' = cond(_n==1, invlogit(`xb3'+`dd')-`Peta1', 0)        // probability to be eta2 type
		gen double `Peta3' = 1-`Peta2'-`Peta1'                                            // probability to be eta3 type

		mlsum `lnf' = log(`Peta1'*`Lf_1'+`Peta2'*`Lf_2' + `Peta3'*`Lf_3')   //if _n==1    // total (unconditional) log-likelihood of all individuals
	}
end

global  id "indiv_id"  // panel id
global  ps "psurv"    // survival prob from agecur to agecur+1  


use   $datadir/UKHLS_healthshock.dta, replace



xtset indiv_id year
sort  indiv_id year

/* I dont have disability insurance so i dont do this but here it is...

* define ever DI receiver
recode DIrcpt (.=-100) (2=-100)
bysort headID1(DIrcpt) : gen everDI = DIrcpt[_N] 
recode DIrcpt (-100=.)


*/
sort indiv_id year
***** prepare data ******  
*quietly {
	* adjust to 2 years age interval & keep only 2 year interval /I dont do this
		
	gen double age2 = age*age
	gen double age3 = age2*age
	gen double healthHp  = F1.healthH    // bad or good health next period
	gen double healthH3p = F1.healthH3

	gen cohortBRY = cond(birthyear<1940,1935, cond(birthyear>1940 & birthyear<=1950, 1945, cond(birthyear>1950 & birthyear<=1960, 1955, ///
	                cond(birthyear>1960 & birthyear<=1970, 1965, cond(birthyear>1970 & birthyear!=., 1975,.)))))
	fvset base 1935 cohortBRY
	
	* gen poor health state
	gen  temp1 = cond(healthH3==3,1,.)
	egen everpoor = total(temp1), by(indiv_id)
	bysort indiv_id everpoor : gen temp2 = 1 if _n==1 & everpoor>=1
	count if temp2==1
	drop  temp*
	count if everpoor>=1
	gen poorHealth = cond(healthH3==3,1, cond(healthH3!=.,0,.))
	
	
	label define poorHealthlab 1 "poor health"
	label value poorHealth poorHealthlab
	fvset base 0 poorHealth 
	count if (poorHealth==1 & healthH==1)
	sort indiv_id year
	gen double poorHealthp = F1.poorHealth    // poor health next period
	
	* construct wealth group (using cross-sectional weight)
	/* I DO NOT DO THIS SINCE I DONT HAVE WEALTH DATA YET*/
	
	
	noisily  local nw = 5
	gen wealthQ`nw'=.
	forvalue aa = 1/12 {
		xtile  Q`nw' = interest [aweight=wght0] if ageint5==`aa', nq(`nw')
		replace  wealthQ`nw' = Q`nw' if ageint5==`aa' //& wealth_fam3!=. 
		drop  Q`nw'
	}
	fvset base 1 wealthQ`nw'
	
	
	
	* create FE tercile (using individual ui and cross-sectional weight)
	bysort indiv_id(year) : gen tag_Obs1=1 if _n==1
	xtile  ui3 = ui  [aweight=wght0]  if tag_Obs1==1, nq(3)
	noisily tab  ui3 [aweight=wght0]  if tag_Obs1==1 
	bysort indiv_id(ui3) : replace  ui3 = ui3[1] 
	sort indiv_id year
	
	* get initial distribution {health, fe3}
	noisily  display("19-25")
	local ifcond = "age>=19 & age<=25 " // & tag_Obs1==1" 
	noisily  tab healthH3          [aweight=wght0] if `ifcond' 
	noisily  tab healthH ui3       [aweight=wght0] if `ifcond', cell nofreq
	noisily  tab healthH ui3       [aweight=wght0] if `ifcond', row nofreq
	
	drop tag_Obs1
	
*}

*******************************************************************
**** Estimate Maximum Likelihood (using fixed weight: wghtFE) *****
*******************************************************************
local agemin = 19
local agemax = 85
keep  if age>=`agemin'  & age<=`agemax'

local  reflag = 5
local  lagmin = 1		


est drop _all

// log using $outdir/Table1_Table2.log, replace

forvalue lagt = `reflag'(-1)`lagmin' {
   
		
	*preserve
//DOES THIS INITIALIZE A WEALTH MATRIX? if so i need to take it out
	local initW   = "0 0 0 0 0"
	
	*quietly {
		***** Compute consecutive lag of Bad and Good health *****
		noisily display("T=`lagt'")
		di "****after noisy display****"
		sort indiv_id year
		if (`lagt'==1) {
			
			gen     tauht = 1 if healthH!=.
			
		}
		else if (`lagt'==2) {
			gen     tauht = 1 if healthH!=L1.healthH & L1.healthH!=.
			replace tauht = 2 if healthH==L1.healthH & L1.healthH!=.
		}
		else if (`lagt'==3) {
			gen     tauht = 1 if healthH!=L1.healthH & L1.healthH!=.
			replace tauht = 2 if healthH==L1.healthH & L1.healthH!=L2.healthH & L2.healthH!=.
			replace tauht = 3 if healthH==L1.healthH & L1.healthH==L2.healthH & L2.healthH!=.
		}
		else if (`lagt'==4) {
			gen     tauht = 1 if healthH!=L1.healthH & L1.healthH!=.
			replace tauht = 2 if healthH==L1.healthH & L1.healthH!=L2.healthH & L2.healthH!=.
			replace tauht = 3 if healthH==L1.healthH & L1.healthH==L2.healthH & L2.healthH!=L3.healthH & L3.healthH!=. 
			replace tauht = 4 if healthH==L1.healthH & L1.healthH==L2.healthH & L2.healthH==L3.healthH & L3.healthH!=. 
		}
		else if (`lagt'==5) {
			di "****tau being created for lag T = 5****"
			gen     tauht = 1 if healthH!=L1.healthH & L1.healthH!=.
			replace tauht = 2 if healthH==L1.healthH & L1.healthH!=L2.healthH & L2.healthH!=.
			replace tauht = 3 if healthH==L1.healthH & L1.healthH==L2.healthH & L2.healthH!=L3.healthH & L3.healthH!=. 
			replace tauht = 4 if healthH==L1.healthH & L1.healthH==L2.healthH & L2.healthH==L3.healthH & L3.healthH!=L4.healthH & L4.healthH!=.
			replace tauht = 5 if healthH==L1.healthH & L1.healthH==L2.healthH & L2.healthH==L3.healthH & L3.healthH==L4.healthH & L4.healthH!=.
			di "****tau created for lag T = 5****"
		}
		
		fvset base 1 tauht  	
		
			
		pause on
		pause

		/* we lose a lot of sample size here, about half (16000)*/
		
		drop if tauht==.
		drop if healthHp==.
		
		
		
		* export and merge with survival probability (estimated from HRS)
		* I don thave survival probs yet could estimate from HRS though
		cd "$survdir"
		noisily  local smax = 3
		merge m:1 ageint2 healthH poorHealth using HRS_PRsurvival.dta
	
		
		cd "$datadir"
		drop if _merge!=3
		drop _merge
		

		
		
		gen psurv =.
		forvalue i = 1/`smax' {
			replace psurv = cond(tauht==`i',psurv`i', cond(tauht>`smax' & tauht!=.,psurv`smax',psurv)) 
		}
		
		sort indiv_id year	
	*}	
	
	bysort indiv_id(year) : gen tag_Obs1=1 if _n==1
	tab healthH ui3 [aweight=wghtFE] if age>=19 & age<=25 & tag_Obs1==1 & tauh!=., col nofreq
	drop tag_Obs1

	
	*** Maximum likelihood estimation ****	
	if (`lagt'==5) {
		local init1 = "0.0563872 0.3988808 -.0002068 -.0028655 0   -8.039628  0 .0752379  .8259081  .527683  .7720454  -1.902561   .0802243 -.0003955 0 -.3914571  -.241504  -1.006626  -1.920926  -3.160139  -.0390794  0 -.2468801  -1.203242  0 1.463351 -1.456916 0   0 -.0018516 -.6194958 -.7485082 -2.34786    0 -.7075066 -1.22599 -.9888237  -.1572297   -.6229429    1.039418  .8761514  1.574786   .8199089  .7147094    4.527345  -1.44721"
		local num_elements : word count `init1'
		display "Number of elements in init1: " `num_elements'
		}
	else if (`lagt'==4) {
		local init1 = "0.0410754  .3667478  -.0000662  -.0026051  0  -7.489549  0  .0714727  .8100455  .7038159  -1.529439  .0511984  -.0001476  0  -.3659227  -.1834267  -1.69093  -2.35133   -.025160  0  -.3368116 -1.37373  0 2.071816 -1.428753  0    0  -.1287474  -.4289286  -.6059185  -1.615753  0  -.4035445  -.7190559  -.7078866  .1179462  -1.256403     1.041878  .8654713  1.469512   .6904142  .7438589   3.784152  -1.638641"
	}
	else if (`lagt'==3) {
		local init1 = ".0047837  .1995778  .0004399 -.001090   0 -3.462979  0  .1293449  .6748344  -1.990339    .0044878  .0005057   0 -.3686884  -1.085505  -2.451367   .0047311   0 -.5085149  -1.188495  0  2.409679 -1.879439 0   0 -.0476497 -.3670531 -.6907869 -1.168707   0 .6120859 .7727794 .9966599 1.777904  -.8117133   1.044391  .8615868  .8828695   .472379  .2971267  1.637419  -2.31758"
		local num_elements : word count `init1'
		display "Number of elements in init1: " `num_elements'
	}
	else if (`lagt'==2) {
		local init1 = ".025102  .1936364  .0003487  -.0009432   0 -2.782818  0  .2882111  -2.79008    .0228017  .0003801   0  -.7695364  -2.98886     .0005205  0  -.5461394  -1.28629  0  2.385837 -1.920989 0   0 -.4588885 -.377868  -.7005325 -1.2803   0 .4345806  .6120905 .6962885 1.48712 -.353521   1.07381  .8653863   .7761271    .6864553  .2646778  1.805973  -2.554666"
		local num_elements : word count `init1'
		display "Number of elements in init1: " `num_elements'
	}
	else if (`lagt'==1) {
		local init1 = ".0436439  .2263482  .0002148 -.0011889   0 -3.136601  0   -3.3115     .0321447  .0003661   0    -3.973014  -.0163349  0 -.6421195  -1.355084  0  1.021862 -2.249955  0   0 -.4685663 -.6026177 -.7588315 -1.264476  0 .0412102 .1282799 .074436 .7627064 1.120506   1.080511  .8766553  .7753692   .7472765  .4099035  2.005649  -2.870534"
		local num_elements : word count `init1'
		display "Number of elements in init1: " `num_elements'
	}
	
di "****YAAAAYYY YOU GOT HEREE*******"	
//this used to include i.wealthQ`nw' in the first parenthesis
		ml model d0 healthML_d0 (xbB: healthHp  poorHealthp  healthH = (c.age c.age2)#i.poorHealth  i.poorHealth  i.tauht) (xbH: c.age c.age2 i.tauht) (xFE: c.age i.ui3 i.healthH#i.poorHealth i.wealthQ`nw' i.cohortBRY) ///
							(b_ddB: )  (b_ddG: )  (b_dd: )  (b_lnetaB1: )  (b_lnetaB3: )  (b_etaH1: )  (b_etaH3: )  [aweight=wghtFE] 
	ml init  `init1', copy
	eststo L`lagt' : ml maximize, difficult tolerance(1e-06)  


	estadd scalar etaB1 =  exp([b_lnetaB1]_b[_cons])
	estadd scalar etaB3 = -exp([b_lnetaB3]_b[_cons])
	estadd scalar d1    =  [xFE]_b[_cons]
	estadd scalar d2    =  [xFE]_b[_cons]+exp([b_dd]_b[_cons])	
// 	esttab using $outdir/Table1_Table2.smcl, compress varwidth(23)  modelwidth(11) stats(N)  mtitles not replace  ///
// 				 drop(1.tauht 0.poorHealth 0.healthH#0.poorHealth 1.healthH#1.poorHealth 1.ui3)
	esttab using $outdir/Table1_Table2.tex, compress varwidth(23)  modelwidth(11) stats(N)  mtitles not replace  ///
				 drop(1.tauht 0.poorHealth 0.healthH#0.poorHealth 1.healthH#1.poorHealth 1.ui3)				 

	nlcom (etaB1_L`lagt': exp([b_lnetaB1]_b[_cons]))  (etaB3_L`lagt': -exp([b_lnetaB3]_b[_cons])) (d2_L`lagt': [xFE]_b[_cons] + exp([b_dd]_b[_cons]))

	
	restore

}

// log close






