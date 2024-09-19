/* Based on De Nardi et al. PSID_documentation1.do we will do the following	
	*****     CONSTRUCT CROSS SECTIONAL PROFILES      *****
	*****   i) use sampleselection1  ii) use wght0 
	***** construct employment profile by health *****
	***** construct % bad health by income group *****
	CONSTRUCT PROFILES by FIXED PRODUCTIVTY      *****
	*****   i) use sampleselection2  ii) use wghtFE 
	***** Estimate fixed effect regression *****
	*** save data for estimating health shock process ***
	*** Save data for estimating labor productivity shock ***
*/

   *******************************************************
   *****     CONSTRUCT CROSS SECTIONAL PROFILES      *****
   *****   i) use sampleselection1  ii) use wght0    *****
   *******************************************************
   	cls
	clear
     
	local dir = "$datadir"
	local savedir "$outdir/UKHLS_excel"
	
	cd  `dir'	
   	
	use UKHLS_sample_trans, clear
	
	keep if sampleselection1 == 1
	drop wght wghtFE
	xtset indiv_id year
	
	***** construct employment profile by health *****
	
	logit emp i.MH3#i.ageint5  i.year [iweight=wght0]  if age>=21 &  age <=70, vsquish    
   
	xtable age MH3 if age>=21 & age<=64 [aweight=wght0],  c(mean   emp) sc filename("`savedir'/TargetedMoments.xlsx") sheet(EmploymentProfile)      format(%9.3f) modify
	putexcel set "`savedir'/TargetedMoments.xlsx",  sheet(EmploymentProfile) modify
	putexcel A1 = "Targted employment moments" 
	xtable ageint2 MH3 if age>=21 & age<=64 [aweight=wght0],  c(semean emp) sc filename("`savedir'/TargetedMoments.xlsx") sheet(SE_EmploymentProfile)   format(%9.3f) modify
	putexcel set "`savedir'/TargetedMoments.xlsx",  sheet(SE_EmploymentProfile)  modify
	putexcel A1 = "SE of targted employment moments"
	xtable ageint5 MH3 if age>=25 & age<=64 [aweight=wght0],  c(mean   emp) sc filename("`savedir'/TargetedMoments.xlsx")    sheet(Figure10_left)   format(%9.3f) modify
	putexcel set "`savedir'/TargetedMoments.xlsx",  sheet(Figure10_left)  modify
	putexcel A1 = "Figure 10 (left) : Employment by health"
	
	**************************************************
    ***** construct % bad health by income group *****
    **************************************************    
    * construct income groups by year
	egen incgroup = xtile(labor_income) if age>=21 & age<=66, by(year ageint10_1) nq(3) weight(wght0)
	putexcel set "`savedir'/UKHLS_Tables.xlsx", sheet(Table9) modify
	xtable ageint10_1 incgroup [aweight=wght0], c(mean badMH) sc  format(%9.3f) noput
	putexcel A1="Table 9 : % unhealthy in each income terciles (UKHLS)"
	putexcel B2= "income terciles"
	putexcel A3=matrix(r(xtable)), names
	
	************************************************************
	************************************************************
	*****     CONSTRUCT PROFILES by FIXED PRODUCTIVTY      *****
	*****   i) use sampleselection2  ii) use wghtFE        *****
	************************************************************
	************************************************************
	use UKHLS_sample_trans, clear
	keep if sampleselection1 == 1
	drop wght
	xtset indiv_id year  	

	local agemin = 17
	
	********************************************
    ***** Estimate fixed effect regression *****
    ********************************************
	set emptycells drop
	xtreg log_labor_inc i.MP2#i.age [aweight=wghtFE] if age>=`agemin', fe vsquish             
	predict loglabincHprof_NoFE   if e(sample), xb    
	predict eit                   if e(sample), e     /* predict only in-sample */                                         
  	predict ui0, u
	
	quietly: reg ui0 i.birthyear [aweight=wghtFE]       /* remove cohort effect from fixed productivity */
	predict ui   if e(sample), res
	gen  ueit  = ui  + eit 
	gen  ueit0 = ui0 + eit
	gen  loglabincHprof_wFE = loglabincHprof_NoFE + ui

    * assign fixed effect for period not working
	bysort indiv_id(ui) : replace ui = ui[1] if ui[1]!=.	
	sort   indiv_id year
	
	
	*======== create FE tercile (using individual ui and cross-sectional weight) =======
	bysort indiv_id(year) : gen tag_2015=1 if year==2015 & wghtFE>0 & wghtFE!=.
	
	xtile  ui3 = ui  [aweight=wght0]  if tag_2015==1 , nq(3)
	tab  ui3 [aweight=wght0]  if tag_2015==1 	
	

	local xx    = "age"
	local agexx = 64	
	
			 
	xtable `xx' MH  if age>=20 & age<=`agexx' & emp==1  [aweight=wght0],  c(mean   loglabincHprof_wFE) sc filename("`savedir'/TargetedMoments.xlsx") sheet(LaborIncProfile)   format(%9.4f) modify	
	putexcel set "`savedir'/TargetedMoments.xlsx", sheet(LaborIncProfile) modify
	putexcel A1 = "Targeted log labor income profile"
	xtable `xx' MH  if age>=20 & age<=`agexx' & emp==1  [aweight=wght0],  c(semean loglabincHprof_wFE) sc filename("`savedir'/TargetedMoments.xlsx") sheet(SE_LaborInc) format(%9.4f) modify
	putexcel set "`savedir'/TargetedMoments.xlsx", sheet(SE_LaborInc) modify
	putexcel A1 = "SE of targeted log labor income profile"
   
   	summarize ui [aweight=wghtFE] if age>=20 & age<=`agexx', detail
	gen   fegroup2 = cond(ui<r(p50),1, cond(ui>=r(p50) & ui!=.,0,.))
	xtile fegroup3 = ui [aweight=wghtFE], nq(3)
	
 	tab age fegroup3 [aweight=wghtFE] if MH==1 & age>=20 & age<=65, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure6") modify	
	putexcel  B1 = "Figure 6 (left) : % bottom terciles of ﬁxed labor productivity (good)"
	putexcel  A2 = "age"
	putexcel  A3 = matrix(yy)  B2 = "bottom 1/3" C2 = "middle 1/3" D2 ="top 1/3"  
	putexcel  B3 = matrix(xx)	
	
	tab age fegroup3 [aweight=wghtFE] if MH==0 & age>=20 & age<=65, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure6") modify	
	putexcel  H1 = "Figure 6 (right) : % bottom terciles of ﬁxed labor productivity (bad)"
	putexcel  G2 = "age"
	putexcel  G3 = matrix(yy)  H2 = "bottom 1/3" I2 = "middle 1/3" J2 ="top 1/3"  
	putexcel  H3 = matrix(xx)
	

	local xx    = "age"
	local agexx = 64
	

	****HERE IS WHERE IT IS ACTUALLY OUTPUT TO EXCEL***
	local hTypes "MH PH MP2"
	foreach hType in `hTypes'{
	preserve
		collapse (mean) ui0 ui fegroup2 [aweight=wghtFE] if age>=20 & age<=`agexx', by(`xx' `hType')
		xtable `xx' `hType' if age>=25, c(mean fegroup2)  filename("`savedir'/UKHLS_De_Nardi_Figures") sheet(Figure3_`hType') format(%9.4f) modify
		putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet(Figure3_`hType') modify
		putexcel A1 = "Figure 3 : Fixed labor productivity by `hType' type"
	restore
	}
	
	
	
	*** save data for estimating shock processes ***
	
	local lastWave = "l"               
 
	cd "$outdir/UKHLS_shocks"
	
	*** save data for estimating health shock process ***
	preserve
	keep indiv_id year age ageint5 ageint10 wghtFE wght`lastWave' wght0 MH MH3 MH4 MH5 ui birthyear //DIrcpt  wealth_fam3 LiteExercisePerWeek HvyExercisePerWeek  AgeFirstSmoking smokingH cigarettH
	sort indiv_id year
// 	save healthML_data2Yinterval_wPoorHealth_R2_2_2.dta, replace
	save UKHLS_healthshock.dta, replace
	restore	
	
	*** Save data for estimating labor productivity shock ***
	preserve
	keep indiv_id year age ageint5 ageint10 wghtFE wght`lastWave' wght0 MH MH3 MH4 MH5 ui birthyear loglabincHprof_NoFE loglabincHprof_wFE  ueit0 ueit  eit  //DIrcpt  wealth_fam3    
	sort indiv_id year
// 	save LabProd_wPoorHealth_R2_2_2.dta, replace
	save UKHLS_laborshock.dta, replace
	restore
	
