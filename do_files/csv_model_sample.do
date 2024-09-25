cls

clear

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files" /*do file folder directory */
global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input/" /*output file folder directory */
global datadir  "$curdir/Data"

local dir = "$datadir"

cd "`dir'"
use "`dir'/UKHLS_sample", clear

*keep age mental_health
cd "$outdir" 
export delimited using "UKHLS_sample.csv", replace
