clear all
cls

set maxvar 20000

global curdir "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model"  /* main folder directory */ 
global dodir  "$curdir/My Code/MH_Model/do_files" /*do file folder directory */
// global outdir   "$curdir/Output" /*output file folder directory */
global datadir  "$curdir/data" //"C:\Users\Ben\3YP_Data_Work\Data"

cd "$curdir"

timer clear 1
timer on 1

// do "$dodir/UKHLS_data_clean.do"  /* clean the data and generate input file for estimation of various statistics */

do "$dodir/UKHLS_sample.do"  /* choose my sample, but leave women in, take them out in quants if you want */

do "$dodir/UKHLS_trans.do" /*calculate some PH and MH transiton probabilities, do some persistance stuff*/

do "$dodir/test.do" 

// do "$dodir/UKHLS_quants.do" /* investigate the quantiles of my continuous mh variable and run some regressions*/

//do $dodir/UKHLS_labor_prod /*estimate labor productivity, and income and employment profiles by heatlh*/


timer off 1

timer list 1


