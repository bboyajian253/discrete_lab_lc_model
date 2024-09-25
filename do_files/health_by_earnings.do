
    // By earnings level (decile, quintile, etc) what is the percentage in bad health state.
    // - how does the model compare to the data?

    // gen earninngs deciles
    gen labor_income_nonmissing = labor_income if labor_income != . & emp == 1
    capture drop earnings_Q10
    xtile earnings_Q10 = labor_income_nonmissing [pweight=wght], nq(10)
    drop labor_income_nonmissing  

    cd "$outdir"

    preserve
    drop if missing(earnings_Q10)
    collapse (mean) badMH, by(earnings_Q10)
    rename badMH mean_badMH
    tempfile mean_bad_MH_by_earnings
    save `mean_bad_MH_by_earnings', replace
    export delimited using "mean_bad_MH_by_earnings.csv", replace
    restore








