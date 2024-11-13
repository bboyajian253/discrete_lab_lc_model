clear all
cls

set maxvar 20000

global curdir "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model" 
global dodir  "$curdir/My Code/MH_Model/do_files" 
// global outdir   "$curdir/Output" 
global datadir  "$curdir/data" 

cd "$curdir"

timer clear 1
timer on 1

// do "$dodir/UKHLS_merge.do" /* merge the data from all waves */

// do "$dodir/UKHLS_clean.do"  /* clean the data and generate input file for estimation of various statistics */

do "$dodir/UKHLS_sample.do"  /* choose my sample */

// do "$dodir/sample_tc_counter.do"  /* choose my sample for the time cost counterfactual */

do "$dodir/UKHLS_trans.do" /*calculate some PH and MH transiton probabilities, do some persistance stuff*/

// do "$dodir/UKHLS_quants.do" /* investigate the quantiles of my continuous mh variable and run some regressions*/

timer off 1

timer list 1


