***BASED ON CODE WITH THE FOLLOWING HEADING***
   * This code compute several statistics from PSID and data for later used to estimate labor productivty shock and health shock process.
   *    updated: Dec 1, 2023   
   
   *  Required input files (stored in 'curdir')
   *    - PSIDsample.dta (automatically created from IND2017ER_rev1.do)
   *
   *  Output files 
   *    (saved in 'curdir')
   *    - PSID_healthshock.dta for estimating health shock parameters
   *    - PSID_laborshock.dta for estimating labor productivity parameters
   *    (saved in 'outdir')
   *    - PSID_Figures.xlsx for lated used in Matlab code to plot figures in the paper and to generate input files from lifecycle model (xxx.in) 
   *    - PSID_Tables.xlsx containing estimates reported in the papers 
   *    - TargetedMoment.xlsx containing estimates later used as targeted moments in life cycle model.

	
	cls
	clear
     
	local dir = "C:\Users\Ben\3YP_Data_Work\Data"
 
	
	cd  `dir'	
	use `dir'/UKHLS_sample, clear
	
	local refyear = 2013
	
*UK CPI all items from FRED. Base year: 2015	
	local cpi2008 = 86.2
	local cpi2009 = 87.9
	local cpi2010 = 90.1
	local cpi2011 = 93.6
	local cpi2012 = 96.0
	local cpi2013 = 98.2
	local cpi2014 = 99.6
	local cpi2015 = 100.0
	local cpi2016 = 101.0
	local cpi2017 = 103.6
	local cpi2018 = 106.0 
	local cpi2019 = 107.8 
	local cpi2020 = 108.9  	
	local cpi2021 = 111.6  
	
	
