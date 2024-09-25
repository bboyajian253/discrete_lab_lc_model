	cls
	clear
     
	local dir = "$datadir"
	local savedir "$dodir/UKHLS_trans_output"
	
	cd  "`dir'"	
	
	// use UKHLS_sample_quants, clear
	use UKHLS_sample, clear

	// pause on
	// pause

	xtset indiv_id age
	
	*define composite health state represenitng both PH and MH states
	gen MP_BB = .
	replace MP_BB = 1 if MH == 0 & PH == 0
	replace MP_BB = 0 if ((MH == 1 & PH == 0)|(MH == 0 & PH == 1)|(MH == 1 & PH == 1))
	label define MP_BBlab 0 "not poor MH and poor PH" 1 "poor MH and poor PH"
	label values MP_BB MP_BBlab
	
	gen MP_BG = .
	replace MP_BG = 1 if MH == 0 & PH == 1
	replace MP_BG = 0 if ((MH == 1 & PH == 0)|(MH == 0 & PH == 0)|(MH == 1 & PH == 1))
	label define MP_BGlab 0 "not poor MH and good PH" 1 "poor MH and good PH"
	label values MP_BG MP_BGlab
	
	gen MP_GG = .
	replace MP_GG = 1 if MH == 1 & PH == 1
	replace MP_GG = 0 if ((MH == 1 & PH == 0)|(MH == 0 & PH == 0)|(MH == 0 & PH == 1))
	label define MP_GGlab 0 "not good MH and good PH" 1 "good MH and good PH"
	label values MP_GG MP_GGlab
	
	gen MP_GB = .
	replace MP_GB = 1 if MH == 1 & PH == 0
	replace MP_GB = 0 if ((MH == 1 & PH == 1)|(MH == 0 & PH == 0)|(MH == 0 & PH == 1))
	label define MP_GBlab 0 "not good MH and poor PH" 1 "good MH and poor PH"
	label values MP_GB MP_GBlab
	
	gen MP2 = .
	replace MP2 = 1 if MP_BB ==1 
	replace MP2 = 2 if MP_BG ==1
	replace MP2 = 3 if MP_GB ==1
	replace MP2 = 4 if MP_GG ==1
	label define MP2lab 1 "Bad Bad" 2 "Bad Good" 3 "Good Bad" 4 "Good Good"
	label values MP2 MP2lab
	
	sort indiv_id age
	
	* 1-years MH transition	
	gen MH_P2G = cond(MH==0 & F1.MH==1,1, cond(MH==0 & F1.MH==0,0,.))	
	gen MH_G2P = cond(MH==1 & F1.MH==0,1, cond(MH==1 & F1.MH==1,0,.)) 
	
	* 1-years MH3 transition
	*starting good
	gen MH3_G2G = cond(MH3==3 & F1.MH3==3,1, cond(MH3==3 & (F1.MH3==2 | F1.MH3==1),0,.))
	gen MH3_G2F = cond(MH3==3 & F1.MH3==2,1, cond(MH3==3 & (F1.MH3==1 | F1.MH3==3),0,.))
	gen MH3_G2P = cond(MH3==3 & F1.MH3==1,1, cond(MH3==3 & (F1.MH3==2 | F1.MH3==3),0,.))
	*starting fair
	gen MH3_F2G = cond(MH3==2 & F1.MH3==3,1, cond(MH3==2 & (F1.MH3==1 | F1.MH3==2),0,.))
	gen MH3_F2F = cond(MH3==2 & F1.MH3==2,1, cond(MH3==2 & (F1.MH3==3 | F1.MH3==1),0,.))
	gen MH3_F2P = cond(MH3==2 & F1.MH3==1,1, cond(MH3==2 & (F1.MH3==3 | F1.MH3==2),0,.))	
	*starting poor
	gen MH3_P2G = cond(MH3==1 & F1.MH3==3,1, cond(MH3==1 & (F1.MH3==2 | F1.MH3==1),0,.))
	gen MH3_P2F = cond(MH3==1 & F1.MH3==2,1, cond(MH3==1 & (F1.MH3==3 | F1.MH3==1),0,.))
	gen MH3_P2P = cond(MH3==1 & F1.MH3==1,1, cond(MH3==1 & (F1.MH3==3 | F1.MH3==2),0,.))
	
	*cd "`savedir'"
	* fraction of fair and poor health
	tab age MH3 [aweight=wght0] if age<=95, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MH") modify	
	putexcel  B1 = "Figure 1 (left) : Fraction of people in bad mental health"
	putexcel  A2 = "age"
	putexcel  A3 = matrix(yy)  B2 = "poor" C2 = "fair" D2 ="good"  
	putexcel  B3 = matrix(xx)

	*1-year binary health state transitions
	tab age MH_P2G [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)	
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MH") modify	
	putexcel  H1 = "Figure 1 (middle) : % transition from bad mental health"
	putexcel  H2 = "age"
	putexcel  H3 = matrix(yy)  I2 = "to poor"  J2 = "to good"  
	putexcel  I3 = matrix(xx)

	tab age MH_G2P [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)		
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MH") modify	
	putexcel  M1 = "Figure 1 (right) : % transition from good mental health"
	putexcel  M2 = "age"
	putexcel  M3 = matrix(yy)  N2 = "to good"  O2 = "to poor" 
	putexcel  N3 = matrix(xx)
	
	local conditions "PH==0 PH==1"
	local i = 0
foreach cond in `conditions'{
		*1-year binary health state transitions
	tab age MH_P2G [aweight=wght0] if age<=89 & `cond', matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)	
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MHPH`i'") modify	
	putexcel  H1 = "Figure 1 (middle) : % transition from bad mental health"
	putexcel  H2 = "age"
	putexcel  H3 = matrix(yy)  I2 = "to poor"  J2 = "to good"  
	putexcel  I3 = matrix(xx)

	tab age MH_G2P [aweight=wght0] if age<=89 & `cond', matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)		
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MHPH`i'") modify	
	putexcel  M1 = "Figure 1 (right) : % transition from good mental health"
	putexcel  M2 = "age"
	putexcel  M3 = matrix(yy)  N2 = "to good"  O2 = "to poor" 
	putexcel  N3 = matrix(xx)
	local ++i

}
	
	* 1-years PH transition	
	gen PH_P2G = cond(PH==0 & F1.PH==1,1, cond(PH==0 & F1.PH==0,0,.))	
	gen PH_G2P = cond(PH==1 & F1.PH==0,1, cond(PH==1 & F1.PH==1,0,.)) 
	
	* 1-years PH3 transition
	*starting good
	gen PH3_G2G = cond(PH3==3 & F1.PH3==3,1, cond(PH3==3 & (F1.PH3==2 | F1.PH3==1),0,.))
	gen PH3_G2F = cond(PH3==3 & F1.PH3==2,1, cond(PH3==3 & (F1.PH3==1 | F1.PH3==3),0,.))
	gen PH3_G2P = cond(PH3==3 & F1.PH3==1,1, cond(PH3==3 & (F1.PH3==2 | F1.PH3==3),0,.))
	*starting fair
	gen PH3_F2G = cond(PH3==2 & F1.PH3==3,1, cond(PH3==2 & (F1.PH3==1 | F1.PH3==2),0,.))
	gen PH3_F2F = cond(PH3==2 & F1.PH3==2,1, cond(PH3==2 & (F1.PH3==3 | F1.PH3==1),0,.))
	gen PH3_F2P = cond(PH3==2 & F1.PH3==1,1, cond(PH3==2 & (F1.PH3==3 | F1.PH3==2),0,.))	
	*starting poor
	gen PH3_P2G = cond(PH3==1 & F1.PH3==3,1, cond(PH3==1 & (F1.PH3==2 | F1.PH3==1),0,.))
	gen PH3_P2F = cond(PH3==1 & F1.PH3==2,1, cond(PH3==1 & (F1.PH3==3 | F1.PH3==1),0,.))
	gen PH3_P2P = cond(PH3==1 & F1.PH3==1,1, cond(PH3==1 & (F1.PH3==3 | F1.PH3==2),0,.))
	
	*cd "`savedir'"
	* fraction of fair and poor health
	tab age PH3 [aweight=wght0] if age<=95, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_PH") modify	
	putexcel  B1 = "Figure 1 (left) : Fraction of people in bad health"
	putexcel  A2 = "age"
	putexcel  A3 = matrix(yy)  B2 = "poor" C2 = "fair" D2 ="good"  
	putexcel  B3 = matrix(xx)

	*1-year binary health state transitions
	tab age PH_P2G [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)	
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_PH") modify	
	putexcel  H1 = "Figure 1 (middle) : % transition from bad"
	putexcel  H2 = "age"
	putexcel  H3 = matrix(yy)  I2 = "to poor"  J2 = "to good"  
	putexcel  I3 = matrix(xx)

	tab age PH_G2P [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)		
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_PH") modify	
	putexcel  M1 = "Figure 1 (right) : % transition from good"
	putexcel  M2 = "age"
	putexcel  M3 = matrix(yy)  N2 = "to good"  O2 = "to poor" 
	putexcel  N3 = matrix(xx)
	
	local conditions "MH==0 MH==1"
	local i = 0
foreach cond in `conditions'{
	*1-year binary physical health state transitions conditional on MH state
	tab age PH_P2G [aweight=wght0] if age<=89 & `cond', matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)	
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_PHMH`i'") modify	
	putexcel  H1 = "Figure 1 (middle) : % physical health transitions from bad `cond'"
	putexcel  H2 = "age"
	putexcel  H3 = matrix(yy)  I2 = "to poor"  J2 = "to good"  
	putexcel  I3 = matrix(xx)

	tab age PH_G2P [aweight=wght0] if age<=89 & `cond', matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)		
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_PHMH`i'") modify	
	putexcel  M1 = "Figure 1 (right) : % transition from good `cond'"
	putexcel  M2 = "age"
	putexcel  M3 = matrix(yy)  N2 = "to good"  O2 = "to poor" 
	putexcel  N3 = matrix(xx)
	local ++i
}	
	****This will be 4 state/2 type  health transitons
/*	
	* 1-years 2 type transitions
	*starting BB
	gen MP_BB2GG = cond(MP_BB == 1 & F1.MP_GG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_BB2GB = cond(MP_BB == 1 & F1.MP_GB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_BB2BG = cond(MP_BB == 1 & F1.MP_BG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_BB2BB = cond(MP_BB == 1 & F1.MP_BB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	*overall transitons variable given starting state
	gen MP_BB2 = .
	replace MP_BB2 = 1 if MP_BB2BB == 1
	replace MP_BB2 = 2 if MP_BB2BG == 1
	replace MP_BB2 = 3 if MP_BB2GB == 1
	replace MP_BB2 = 4 if MP_BB2GG == 1
	label define MP_BB2lab 1 "to Bad Bad" 2 "to Bad Good" 3 "to Good Bad" 4 "to Good Good"
	label values MP_BB2 MP_BB2lab
	
	
	*starting BG
	gen MP_BG2GG = cond(MP_BG == 1 & F1.MP_GG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_BG2GB = cond(MP_BG == 1 & F1.MP_GB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_BG2BG = cond(MP_BG == 1 & F1.MP_BG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_BG2BB = cond(MP_BG == 1 & F1.MP_BB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	*overall transitons variable given starting state
	gen MP_BG2 = .
	replace MP_BG2 = 1 if MP_BG2BB == 1
	replace MP_BG2 = 2 if MP_BG2BG == 1
	replace MP_BG2 = 3 if MP_BG2GB == 1
	replace MP_BG2 = 4 if MP_BG2GG == 1
	label define MP_BG2lab 1 "to Bad Bad" 2 "to Bad Good" 3 "to Good Bad" 4 "to Good Good"
	label values MP_BG2 MP_BG2lab
	
	*starting GB
	gen MP_GB2GG = cond(MP_GB == 1 & F1.MP_GG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_GB2GB = cond(MP_GB == 1 & F1.MP_GB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_GB2BG = cond(MP_GB == 1 & F1.MP_BG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_GB2BB = cond(MP_GB == 1 & F1.MP_BB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	*overall transitons variable given starting state
	gen MP_GB2 = .
	replace MP_GB2 = 1 if MP_GB2BB == 1
	replace MP_GB2 = 2 if MP_GB2BG == 1
	replace MP_GB2 = 3 if MP_GB2GB == 1
	replace MP_GB2 = 4 if MP_GB2GG == 1
	label define MP_GB2lab 1 "to Bad Bad" 2 "to Bad Good" 3 "to Good Bad" 4 "to Good Good"
	label values MP_GB2 MP_GB2lab
	
	*starting GG
	gen MP_GG2GG = cond(MP_GG == 1 & F1.MP_GG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_GG2GB = cond(MP_GG == 1 & F1.MP_GB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_GG2BG = cond(MP_GG == 1 & F1.MP_BG == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	gen MP_GG2BB = cond(MP_GG == 1 & F1.MP_BB == 1, 1, cond(F1.MH == . | F1.PH == ., ., 0))
	*overall transitons variable given starting state
	gen MP_GG2 = .
	replace MP_GG2 = 1 if MP_GG2BB == 1
	replace MP_GG2 = 2 if MP_GG2BG == 1
	replace MP_GG2 = 3 if MP_GG2GB == 1
	replace MP_GG2 = 4 if MP_GG2GG == 1
	label define MP_GG2lab 1 "to Bad Bad" 2 "to Bad Good" 3 "to Good Bad" 4 "to Good Good"
	label values MP_GG2 MP_GG2lab
	

	*cd "`savedir'"
	* shares in each composite health state
	tab age MP2 [aweight=wght0] if age<=95, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MP") modify	
	putexcel  B1 = "Figure 1 (a) : Fraction of people in composite health state"
	putexcel  A2 = "age"
	putexcel  A3 = matrix(yy)  B2 = "Bad Bad" C2 = "Bad Good" D2 ="Good Bad"  E2 = "Good Good"
	putexcel  B3 = matrix(xx)

	*1-year binary health state transitions
	local startStates "BB BG GB GG"
	local colNames "H I J K L M N O P Q R S T U V W X Y Z AA AB AC"
	tokenize "`colNames'"
	local i = 1
	
foreach state in `startStates'{
	tab age MP_`state'2 [aweight=wght0] if age<=89, matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)	
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure1_MP") modify
	
	putexcel  ``i''1 = "Figure 1 (`state') : % transition from `state'"
	putexcel  ``i''2 = "age"
	
	putexcel  ``i''3 = matrix(yy)  
	local ++i
	putexcel ``i''2 = "to Bad Bad"  ``i''3 = matrix(xx)
	local ++i
	putexcel ``i''2 = "to Bad Good"
	local ++i
	putexcel ``i''2 = "to Good Bad"
	local ++i
	putexcel ``i''2 = "to Good Good"
	local ++i
}

	* define individual tag
   sort indiv_id year
   by indiv_id: gen IDtag = 1 if _n==1    
 
 	gen healthH = MP2
	* construct hazard rate
	*quietly {
	   * define sample being observed for 2-7 consecutive periods
		
		sort indiv_id year 

******NEED TO DO THE FOLLOWING******
 * define sample being observed for 2-7 consecutive periods**
 *uses code kinda like the one below
 ***********************************
 quietly{
 	   * define sample being observed for 2-7 consecutive periods for the composit health type which I should use for both conditional
	   *an unconditional probabilities 
	   gen temp = 1   if MP2 !=.
	   sort indiv_id year
	   gen TR2 = .
		replace TR2 = 1 if F1.temp==1 & temp==1
	   gen TR3 = .
		replace TR3 = 1 if F1.temp==1 & L1.temp==1 & temp==1
	   gen TR4 = .
		replace TR4 = 1 if F1.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
	   gen TR5 = .
		replace TR5 = 1 if F1.temp==1 & L3.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
	   gen TR6 = .
		replace TR6 = 1 if F1.temp==1 & L4.temp==1 & L3.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
	   gen TR7 = .
		replace TR7 = 1 if F1.temp==1 & L5.temp==1 & L4.temp==1 & L3.temp==1 & L2.temp==1 & L1.temp==1 & temp==1
	   drop temp  
		
		
	* construct hazard rate
	local states "BB BG GB GG"
	tokenize "`states'"

	forvalues i = 1/4{
		forvalues j = 1/4{	
			gen 	haz``i''2``j''1 = 0 if TR2 == 1 & MP2 ==`i'  //is Bad Bad
			replace haz``i''2``j''1 = 1 if TR2 == 1 & MP2 ==`i' & F1.MP2 ==`j'
			gen 	haz``i''2``j''2 = 0 if TR3 == 1 & MP2 ==`i' & L1.MP2 ==`i'
			replace haz``i''2``j''2 = 1 if TR3 == 1 & MP2 ==`i' & L1.MP2 ==`i' & F1.MP2 ==`j'
			gen 	haz``i''2``j''3 = 0 if TR4 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i'
			replace haz``i''2``j''3 = 1 if TR4 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & F1.MP2 ==`j'
			gen 	haz``i''2``j''4 = 0 if TR5 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & L3.MP2 ==`i'
			replace haz``i''2``j''4 = 1 if TR5 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & L3.MP2 ==`i' & F1.MP2 ==`j'
			gen 	haz``i''2``j''5 = 0 if TR6 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & L3.MP2 ==`i' & L4.MP2 ==`i'
			replace haz``i''2``j''5 = 1 if TR6 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & L3.MP2 ==`i' & L4.MP2 ==`i' & F1.MP2 ==`j'
			gen 	haz``i''2``j''6 = 0 if TR7 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & L3.MP2 ==`i' & L4.MP2 ==`i' & L5.MP2 ==`i'
			replace haz``i''2``j''6 = 1 if TR7 == 1 & MP2 ==`i' & L1.MP2 ==`i' & L2.MP2 ==`i' & L3.MP2 ==`i' & L4.MP2 ==`i' & L5.MP2 ==`i' & F1.MP2 ==`j'		
		}
	}
}

local states = "BB BG GB GG"
local cols "A B C D E F I J K L M N Q R S T U V Y Z AA AB AC AD"	
tokenize "`cols'"

**also do it unconditionla by age (i.e. not conditional on age)
local col = 1
foreach stateCur in `states'{
	local r = 1
	foreach stateFut in `states'{
		local c = `col'
		* duration profile for health status
		putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure2_MP_noage") modify
		forvalues i = 1/5{
		sum haz`stateCur'2`stateFut'`i' [w=wghtFE], detail
		local meanhaz`stateCur'2`stateFut'`i' = r(mean)
		}
		*local meanhaz`stateCur'2`stateFut'`i' = wtmean(haz`stateCur'2`stateFut'`i'), weight(wghtFE)
		*xtable, c(meanhaz`stateCur'2`stateFut'1 meanhaz`stateCur'2`stateFut'2 meanhaz`stateCur'2`stateFut'3 meanhaz`stateCur'2`stateFut'4 meanhaz`stateCur'2`stateFut'5)  noput
		matrix mytable = (`meanhaz`stateCur'2`stateFut'1', `meanhaz`stateCur'2`stateFut'2', `meanhaz`stateCur'2`stateFut'3', `meanhaz`stateCur'2`stateFut'4', `meanhaz`stateCur'2`stateFut'5')
		
		local r = (`r'-1)*6 + 1 //move excel row index so tables dont write on top of each other
		putexcel ``c''`r' = "Figure 2 : Percentage of transitions from `stateCur' to `stateFut' conditional on being in `stateCur' health (UKHLS)"
		local ++r
		putexcel ``c''`r' = matrix(mytable), names
		local ++c
		putexcel ``c''`r' = ">=1"   
		local ++c
		putexcel ``c''`r' = ">=2"    
		local ++c
		putexcel ``c''`r' = ">=3"
		local ++c
		putexcel ``c''`r' = ">=4"
		local ++c
		putexcel ``c''`r' = ">=5"
		local r = (`r'-2)/6 + 1 //adjust r back to og value
		local ++r 			//incremnet counter for next iteration		
	}	
	local col = `col' + 6
}

local col = 1
foreach stateCur in `states'{
	local r = 1
	foreach stateFut in `states'{
		local c = `col'
		* duration profile for health status
		putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure2_MP") modify
		xtable ageint20 [aweight=wghtFE], c(mean haz`stateCur'2`stateFut'1 mean haz`stateCur'2`stateFut'2 mean haz`stateCur'2`stateFut'3 mean haz`stateCur'2`stateFut'4 mean haz`stateCur'2`stateFut'5)  noput
		local r = (`r'-1)*6 + 1 //move excel row index so tables dont write on top of each other
		putexcel ``c''`r' = "Figure 2 : Percentage of transitions from `stateCur' to `stateFut' conditional on being in `stateCur' health (UKHLS)"
		local ++r
		putexcel ``c''`r' = matrix(r(xtable)), names
		local ++c
		putexcel ``c''`r' = ">=1"   
		local ++c
		putexcel ``c''`r' = ">=2"    
		local ++c
		putexcel ``c''`r' = ">=3"
		local ++c
		putexcel ``c''`r' = ">=4"
		local ++c
		putexcel ``c''`r' = ">=5"
		local r = (`r'-2)/6 + 1 //adjust r back to og value
		local ++r 			//incremnet counter for next iteration	
	}	
	local col = `col' + 6
}
/*
	xtable ageint20 [aweight=wghtFE], c(mean hazrH1 mean hazrH2 mean hazrH3 mean hazrH4 mean hazrH5)   noput  
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad health conditional on being in good health (UKHLS)"
	putexcel A12 = matrix(r(xtable)), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
*/	
	
	local healthTypes "MH PH"
	
foreach hType in `healthTypes'{
	di "*****BELOW IS THE HTYPE OUTPUT*******"
	di `hType'
	capture drop healthH	
	gen healthH = `hType'
	
	if "`hType'" == "MH"{ //for some reason the local MH seems to evaluate to 1 in this context see di MH
		local oType PH
	} 
	else if "`hType'" == "PH"{ //for some reason the local PH seems to evaluate to 0 in this context see di PH
		local oType MH
	}
	local sheetName1 "Figure2_bad`oType'_noage"
	local sheetName2 "Figure2_bad`oType'"
	local sheetName3 "Figure2_good`oType'_noage"
	local sheetName4 "Figure2_good`oType'"
	
	*pause
	* construct hazard rate
	
		sort indiv_id year 
		
		* construct hazard rate
		gen     hazrB1 = 0 if TR2==1 & healthH==0 & bad`oType' == 1
		replace hazrB1 = 1 if TR2==1 & healthH==0 & F1.healthH==1 & bad`oType' == 1
		gen     hazrB2 = 0 if TR3==1 & healthH==0 & L1.healthH==0 & bad`oType' == 1
		replace hazrB2 = 1 if TR3==1 & healthH==0 & L1.healthH==0 & F1.healthH==1 & bad`oType' == 1
		gen     hazrB3 = 0 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & bad`oType' == 1
		replace hazrB3 = 1 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & F1.healthH==1 & bad`oType' == 1
		gen     hazrB4 = 0 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & bad`oType' == 1
		replace hazrB4 = 1 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & F1.healthH==1 & bad`oType' == 1
		gen     hazrB5 = 0 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & bad`oType' == 1 
		replace hazrB5 = 1 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & F1.healthH==1 & bad`oType' == 1
		gen     hazrB6 = 0 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 & bad`oType' == 1
		replace hazrB6 = 1 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 & F1.healthH==1 & bad`oType' == 1
		gen     hazrH1 = 0 if TR2==1 & healthH==1 & bad`oType' == 1
		replace hazrH1 = 1 if TR2==1 & healthH==1 & F1.healthH==0 & bad`oType' == 1
		gen     hazrH2 = 0 if TR3==1 & healthH==1 & L1.healthH==1 & bad`oType' == 1
		replace hazrH2 = 1 if TR3==1 & healthH==1 & L1.healthH==1 & F1.healthH==0 & bad`oType' == 1
		gen     hazrH3 = 0 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & bad`oType' == 1
		replace hazrH3 = 1 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & F1.healthH==0 & bad`oType' == 1
		gen     hazrH4 = 0 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & bad`oType' == 1
		replace hazrH4 = 1 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & F1.healthH==0 & bad`oType' == 1
		gen     hazrH5 = 0 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & bad`oType' == 1
		replace hazrH5 = 1 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & F1.healthH==0 & bad`oType' == 1  
		gen     hazrH6 = 0 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 & bad`oType' == 1
		replace hazrH6 = 1 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 & F1.healthH==0  & bad`oType' == 1 
	
   
	
	* duration profile for mental health status conditional on bad PH
	*unconditional
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet(`sheetName1') modify
	forvalues i = 1/5{
		sum hazrB`i' [w=wghtFE], detail
		local meanHazrB`i' = r(mean)
	}
	matrix myTable = (`meanHazrB1', `meanHazrB2', `meanHazrB3', `meanHazrB4', `meanHazrB5')
	putexcel A1 = "Figure 2 : Percentage of transitions from bad to good `hType' conditional on being in bad `oType' all ages(UKHLS)"
	putexcel A2 = matrix(myTable), names
	putexcel B2 = ">=1"   C2 = ">=2"    D2 = ">=3"    E2 = ">=4"    F2 = ">=5"
	
	forvalues i = 1/5{
		sum hazrH`i' [w=wghtFE], detail
		local meanHazrH`i' = r(mean)
	}
	matrix myTable = (`meanHazrH1', `meanHazrH2', `meanHazrH3', `meanHazrH4', `meanHazrH5')
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad `hType' conditional on being in bad `oType' all ages(UKHLS)"
	putexcel A12 = matrix(myTable), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
	
	*by age group
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet(`sheetName2') modify
	xtable ageint20 [aweight=wghtFE], c(mean hazrB1 mean hazrB2 mean hazrB3 mean hazrB4 mean hazrB5)  noput
	putexcel A1 = "Figure 2 : Percentage of transitions from bad to good `hType' conditional on being in bad `oType' (UKHLS)"
	putexcel A2 = matrix(r(xtable)), names
	putexcel B2 = ">=1"   C2 = ">=2"    D2 = ">=3"    E2 = ">=4"    F2 = ">=5"

	xtable ageint20 [aweight=wghtFE], c(mean hazrH1 mean hazrH2 mean hazrH3 mean hazrH4 mean hazrH5)   noput  
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad `hType' conditional on being in good `oType' (UKHLS)"
	putexcel A12 = matrix(r(xtable)), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
	
	drop hazr*
	* construct hazard rate
	quietly {
	   * define sample being observed for 2-7 consecutive periods
		
		sort indiv_id year 
		
		
		gen     hazrB1 = 0 if TR2==1 & healthH==0 & bad`oType' == 0
		replace hazrB1 = 1 if TR2==1 & healthH==0 & F1.healthH==1 & bad`oType' == 0
		gen     hazrB2 = 0 if TR3==1 & healthH==0 & L1.healthH==0 & bad`oType' == 0
		replace hazrB2 = 1 if TR3==1 & healthH==0 & L1.healthH==0 & F1.healthH==1 & bad`oType' == 0
		gen     hazrB3 = 0 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & bad`oType' == 0
		replace hazrB3 = 1 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & F1.healthH==1 & bad`oType' == 0
		gen     hazrB4 = 0 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & bad`oType' == 0
		replace hazrB4 = 1 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & F1.healthH==1 & bad`oType' == 0
		gen     hazrB5 = 0 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & bad`oType' == 0 
		replace hazrB5 = 1 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & F1.healthH==1 & bad`oType' == 0
		gen     hazrB6 = 0 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 & bad`oType' == 0
		replace hazrB6 = 1 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 & F1.healthH==1 & bad`oType' == 0
		gen     hazrH1 = 0 if TR2==1 & healthH==1 & bad`oType' == 0
		replace hazrH1 = 1 if TR2==1 & healthH==1 & F1.healthH==0 & bad`oType' == 0
		gen     hazrH2 = 0 if TR3==1 & healthH==1 & L1.healthH==1 & bad`oType' == 0
		replace hazrH2 = 1 if TR3==1 & healthH==1 & L1.healthH==1 & F1.healthH==0 & bad`oType' == 0
		gen     hazrH3 = 0 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & bad`oType' == 0
		replace hazrH3 = 1 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & F1.healthH==0 & bad`oType' == 0
		gen     hazrH4 = 0 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & bad`oType' == 0
		replace hazrH4 = 1 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & F1.healthH==0 & bad`oType' == 0
		gen     hazrH5 = 0 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & bad`oType' == 0
		replace hazrH5 = 1 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & F1.healthH==0 & bad`oType' == 0  
		gen     hazrH6 = 0 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 & bad`oType' == 0
		replace hazrH6 = 1 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 & F1.healthH==0  & bad`oType' == 0 
   }
	
	* duration profile for mental health status conditional on good PH
	*unconditional
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet(`sheetName3') modify
	forvalues i = 1/5{
		sum hazrB`i' [w=wghtFE], detail
		local meanHazrB`i' = r(mean)
	}
	matrix myTable = (`meanHazrB1', `meanHazrB2', `meanHazrB3', `meanHazrB4', `meanHazrB5')
	putexcel A1 = "Figure 2 : Percentage of transitions from bad to good `hType' conditional on being in good `oType' all ages(UKHLS)"
	putexcel A2 = matrix(myTable), names
	putexcel B2 = ">=1"   C2 = ">=2"    D2 = ">=3"    E2 = ">=4"    F2 = ">=5"
	
	forvalues i = 1/5{
		sum hazrH`i' [w=wghtFE], detail
		local meanHazrH`i' = r(mean)
	}
	matrix myTable = (`meanHazrH1', `meanHazrH2', `meanHazrH3', `meanHazrH4', `meanHazrH5')
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad `hType' conditional on being in good `oType' all ages(UKHLS)"
	putexcel A12 = matrix(myTable), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
	
	*by age group
	* duration profile for health status
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet(`sheetName4') modify
	xtable ageint20 [aweight=wghtFE], c(mean hazrB1 mean hazrB2 mean hazrB3 mean hazrB4 mean hazrB5)  noput
	putexcel A1 = "Figure 2 : Percentage of transitions from bad to good `hType' conditional on being in bad `oType'(UKHLS)"
	putexcel A2 = matrix(r(xtable)), names
	putexcel B2 = ">=1"   C2 = ">=2"    D2 = ">=3"    E2 = ">=4"    F2 = ">=5"

	xtable ageint20 [aweight=wghtFE], c(mean hazrH1 mean hazrH2 mean hazrH3 mean hazrH4 mean hazrH5)   noput  
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad `hType' conditional on being in good `oType' (UKHLS)"
	putexcel A12 = matrix(r(xtable)), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
	
	drop healthH
	drop hazr*
}	
	*record simple averages of MH and PH to graph in Matlab
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Avg_MHPH_byage") modify
	xtable age [aweight=wghtFE], c(mean mental_health mean physical_health)   noput 
	putexcel A1 = "Figure i: Average Mental and Physical Health by Age"
	putexcel A2 = matrix(r(xtable)), names
	
	local hTypes "MH PH"
	foreach hType in `hTypes'{
		sort indiv_id year 
		gen healthH = `hType'
		* construct hazard rate
		gen     hazrB1 = 0 if TR2==1 & healthH==0 
		replace hazrB1 = 1 if TR2==1 & healthH==0 & F1.healthH==1 
		gen     hazrB2 = 0 if TR3==1 & healthH==0 & L1.healthH==0
		replace hazrB2 = 1 if TR3==1 & healthH==0 & L1.healthH==0 & F1.healthH==1 
		gen     hazrB3 = 0 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 
		replace hazrB3 = 1 if TR4==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & F1.healthH==1 
		gen     hazrB4 = 0 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 
		replace hazrB4 = 1 if TR5==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & F1.healthH==1 
		gen     hazrB5 = 0 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 
		replace hazrB5 = 1 if TR6==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & F1.healthH==1 
		gen     hazrB6 = 0 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 
		replace hazrB6 = 1 if TR7==1 & healthH==0 & L1.healthH==0 & L2.healthH==0 & L3.healthH==0 & L4.healthH==0 & L5.healthH==0 & F1.healthH==1 
		gen     hazrH1 = 0 if TR2==1 & healthH==1  
		replace hazrH1 = 1 if TR2==1 & healthH==1 & F1.healthH==0 
		gen     hazrH2 = 0 if TR3==1 & healthH==1 & L1.healthH==1 
		replace hazrH2 = 1 if TR3==1 & healthH==1 & L1.healthH==1 & F1.healthH==0 
		gen     hazrH3 = 0 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 
		replace hazrH3 = 1 if TR4==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & F1.healthH==0 
		gen     hazrH4 = 0 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 
		replace hazrH4 = 1 if TR5==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & F1.healthH==0 
		gen     hazrH5 = 0 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 
		replace hazrH5 = 1 if TR6==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & F1.healthH==0  
		gen     hazrH6 = 0 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 
		replace hazrH6 = 1 if TR7==1 & healthH==1 & L1.healthH==1 & L2.healthH==1 & L3.healthH==1 & L4.healthH==1 & L5.healthH==1 & F1.healthH==0  
	
   
	
	* duration profile for mental health status conditional on bad PH
	*unconditional
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure2_`hType'_noage") modify
	forvalues i = 1/5{
		sum hazrB`i' [w=wghtFE], detail
		local meanHazrB`i' = r(mean)
	}
	matrix myTable = (`meanHazrB1', `meanHazrB2', `meanHazrB3', `meanHazrB4', `meanHazrB5')
	putexcel A1 = "Figure 2 : Percentage of transitions from bad to good `hType' all ages(UKHLS)"
	putexcel A2 = matrix(myTable), names
	putexcel B2 = ">=1"   C2 = ">=2"    D2 = ">=3"    E2 = ">=4"    F2 = ">=5"
	
	forvalues i = 1/5{
		sum hazrH`i' [w=wghtFE], detail
		local meanHazrH`i' = r(mean)
	}
	matrix myTable = (`meanHazrH1', `meanHazrH2', `meanHazrH3', `meanHazrH4', `meanHazrH5')
	putexcel A11 = "Figure 2 : Percentage of transitions from good to bad `hType' all ages(UKHLS)"
	putexcel A12 = matrix(myTable), names
	putexcel B12 = ">=1"   C12 = ">=2"    D12 = ">=3"    E12 = ">=4"    F12 = ">=5"
	
	drop healthH
	drop hazr*
	}
	*generate the number of periods in bad physical health now that you are old
	preserve
	capture drop healthH
	gen healthH = MH
	
	* get balance panel
	keep if age>=55 & age<=65		
	egen nonmissinghealth = count(healthH==.), by(indiv_id)
	keep if nonmissinghealth==6 /*why 6??*/
	count
	sort indiv_id year
	bysort indiv_id : gen  health55 = healthH[1]   // health at 55
	keep if age>=56 & age<=65
	bysort indiv_id : egen tempH55 = total(badPH) if health55==1   // unhealthy year if healthy at 55
	bysort indiv_id : egen tempU55 = total(badPH) if health55==0   // unhealthy year if unhealthy at 55
	gen UHyrs5765 = cond(health55==1,tempH55, cond(health55==0,tempU55,.))
	keep if age==56
		
	putexcel set "`savedir'/UKHLS_De_Nardi_Figures", sheet("Figure5") modify
	noisily tab health55 UHyrs5765 [aweight=wghtFE], matcell(xx) matrow(yy) matcol(zz)
	mata : xx = st_matrix("xx")
	mata : xx = xx :/rowsum(xx)
	mata : st_matrix("xx",xx)
	putexcel  B1 = "No. of periods in bad health (aged 56-65)"
	putexcel  A2 = "health at 55"
	putexcel  A3 = matrix(yy)  B2 = matrix(zz)  B3 = matrix(xx)
		
	restore 
	*/
	save UKHLS_sample_trans, replace
