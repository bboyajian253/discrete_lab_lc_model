
trans_name = "MH_trans_by_MH_clust_k2_age"
dis_trans_path = trans_path + trans_name + ".csv" 
myPars.H_trans = io.read_and_shape_H_trans_full(myPars, dis_trans_path)
plot_aggregates.plot_H_trans_H_type(myPars, outpath, trans_name) 

importlib.reload(tables)
r2_path = main_path + "/input/MH_trans/" + "MH_type_reg_r2.csv"
my_r2_mat = tb.read_matrix_from_csv(r2_path, column_index = 0)
print(my_r2_mat)
tables.table_r2_by_type_alg(myPars, my_r2_mat, outpath, "tab_MH_type_reg_r2")

importlib.reload(tables)
trans_name = "MH_trans_by_MH_clust_50p"
dis_trans_path = trans_path + trans_name + ".csv" 
trans_50p = tb.read_matrix_from_csv(dis_trans_path, column_index = 0).reshape(2,2,2)
# print(trans_50p)

trans_name = "MH_trans_by_MH_clust_k2"
dis_trans_path = trans_path + trans_name + ".csv" 
trans_k2 = tb.read_matrix_from_csv(dis_trans_path, column_index=0).reshape(2,2,2)
# print(trans_k2)

tables.table_H_trans_by_type_alg(myPars, trans_50p, trans_k2, outpath)

trans_name = "MH_trans_by_MH_clust_k2_age"
dis_trans_path = trans_path + trans_name + ".csv" 
H_trans1 = io.read_and_shape_H_trans_full(myPars, dis_trans_path)

trans_name = "MH_trans_by_MH_clust_50p_age"
dis_trans_path = trans_path + trans_name + ".csv" 
H_trans2 = io.read_and_shape_H_trans_full(myPars, dis_trans_path)

myPars1 = Pars(main_path, J=51, H_trans=H_trans1)
myPars2 = Pars(main_path, J=51, H_trans=H_trans2)

plot_aggregates.plot_H_trans_H_type_alg(myPars1, myPars2, path=outpath)