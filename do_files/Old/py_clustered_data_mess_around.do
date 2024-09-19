cls

clear

global curdir   "C:/Users/Ben/3YP_Data_Work"  /* main folder directory */ 
global dodir    "$curdir/Do_Files" /*do file folder directory */
global outdir   "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input/" /*output file folder directory */
global datadir  "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/output/"

local dir = "$datadir"

cd "`dir'"
import delimited "`dir'/MH_data_hist_clusters.csv", clear
