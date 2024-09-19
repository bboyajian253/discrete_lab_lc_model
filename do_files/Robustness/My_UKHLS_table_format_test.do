local savedir "$outdir/UKHLS_quant"
cd `savedir'

filefilter reg_results_Q5_specEd.tex reg_results_Q5_specEd_hline.tex, from(\BSmidrule) to(\BShline) replace
