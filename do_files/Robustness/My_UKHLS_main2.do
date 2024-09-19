****This file is base don the following from De Nardi et al 2023 ReStud***


**** This program estimates statitsics from PSID ****
*    updated: Dec 1, 2023
*
*    input file:
*           (PSID data)
*        => Downloaded PSID data (stored in 'datadir')
*            - 1984-2017 family files (FAMxxx.txt, FAMxxxER.txt)
*            - cross-year individual file (IND2021ER.txt) 
*            - wealth files (WLTH1984, WLTH1984.txt, WLTH1994.txt, WLTH1999.txt, WLTH2001.txt, WLTH2003.txt, WLTH2005.txt, WLTH2007.txt)
*
*           (HRS data)
*        => HRS_PRsurvival.dta from estimated from HRS (stored in 'curdir')
*
*           (data extraction codes)
*        => Stata codes (FAMxxx_rev1.do, IND2021ER_rev1.do.WLTHxxxx_rev1.do) to extract variables from downloaded PSID data (stored in 'subdir')
*
*           (data analysis code)
*        => PSID_documentation1.do, PSID_documentation2.do, PSID_estLaborShock.do, PSID_EstHealthShock.do, PSID_EstHealthShock_exercise.do (stored in 'curdir')
*  
*
*    output files (saved in 'outdir')
*        => PSID_Figures.xlsx for lated used in Matlab code to plot figures in the paper and to generate input files from lifecycle model (xxx.in) 
*        => PSID_Tables.xlsx, Table1_Table2.tex, Table6_TableC9.tex containing estimates reported in the papers 
*        => Parameters_LaborIncShock.txt for estimates of annual labor productivity shock
*        => TargetedMoment.xlsx containing estimates later used as targeted moments in life cycle model.
*        => FigureB2.eps
*
*
*    Before executing, users need to update the directory part to the main folder 'curdir'. 

clear all
cls

set maxvar 20000

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files/Robustness" /*do file folder directory */
global outdir   "$curdir/Output2" /*output file folder directory */
global datadir  "$curdir/Data" //"C:\Users\Ben\3YP_Data_Work\Data"
global survdir  "$curdir/HRS/HRSout" //

global subdir   "$curdir/PSIDsubprogram"


cd $curdir

timer clear 1

timer on 1
    


do $dodir/My_UKHLS_data_clean    /* clean the data and generate input file for estiamtion of various statistics */


do $dodir/My_UKHLS_sample.do      /* choose my sample, buit leave men in, take them out in quants if you want */

do $dodir/My_UKHLS_quants.do /* investigate the quantiles of my continuous mh variable*/

do $dodir/My_UKHLS_probs.do /*calculate some PH and MH transiton probabilities, do some persistance stuff*/

*do $dodir/My_UKHLS_graphs.do //generate some graohs of MH and PH mostly by age

do $dodir/My_UKHLS_labor_prod /*estimate labor productivity, and income and employment profiles by heatlh*/

*do $dodir/My_UKHLS_table_format_test.do

timer off 1

timer list 1


