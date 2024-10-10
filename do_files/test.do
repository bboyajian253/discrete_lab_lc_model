tab age MH_clust_k2  [aweight=wght0], matcell(xx) matrow(yy) matcol(zz)

mata : xx = st_matrix("xx")
mata : yy = st_matrix("yy")
mata : xx = xx :/ rowsum(xx)
mata : st_matrix("xx", xx)
mata : xx
