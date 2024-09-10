import numpy as np
import main
import pars_shocks as ps
import importlib
importlib.reload(main)
import plot_inequality as plot_ineq
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import my_toolbox as tb

if __name__ == "__main__":
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    trans_path = main_path + "input/k-means/MH_trans_by_MH_clust_age.csv"
    of_name = None
    myPars_bench, myShocks_bench, sols_bench, sims_bench = main.main_io(main_path, out_folder_name = of_name, 
                                                                        H_trans_path = trans_path, output_flag = False)

    myPars_no_cost = main.pars_factory(main_path, H_trans_path = trans_path)
    myPars_no_cost.phi_H = 0.0
    myPars_no_cost, myShocks_no_cost, sols_no_cost, sims_no_cost = main.main_io(main_path, out_folder_name = of_name, 
                                                                                H_trans_path = trans_path,
                                                                                myPars = myPars_no_cost, myShocks=myShocks_bench, 
                                                                                output_flag = False)

    myPars_no_wH = main.pars_factory(main_path, H_trans_path = trans_path)
    myPars_no_wH.wH_coeff = 0.0
    myPars_no_wH, myShocks_no_wH, sols_no_wH, sims_no_wH = main.main_io(main_path, out_folder_name = of_name, H_trans_path = trans_path,
                                                                        myPars = myPars_no_wH, myShocks=myShocks_bench, 
                                                                        output_flag = False, do_wH_calib = False)

    myPars_all_high_H_types = main.pars_factory(main_path, H_trans_path = trans_path)
    myPars_all_high_H_types.H_type_perm_weights = np.array([0.0, 1.0])
    (myPars_all_high_H_types, myShocks_all_high_H_types, 
        sols_all_high_H_types, sims_all_high_H_types) = main.main_io(main_path, out_folder_name = of_name, H_trans_path = trans_path,
                                                                        myPars = myPars_all_high_H_types, myShocks=myShocks_bench, 
                                                                                                        output_flag = False)

    myPars_no_MH = main.pars_factory(main_path, H_trans_path = trans_path)
    myPars_no_MH.phi_H = 0.0
    myPars_no_MH.wH_coeff = 0.0
    myPars_no_MH, myShocks_no_MH, sols_no_MH, sims_no_MH = main.main_io(main_path, out_folder_name = of_name, H_trans_path = trans_path,
                                                                        myPars = myPars_no_MH, myShocks=myShocks_bench,
                                                                        output_flag = False, do_wH_calib = False)


    print("myPars_bench")
    print(f"phi_H: {myPars_bench.phi_H}, wH_coeff: {myPars_bench.wH_coeff}, H_type_perm_weights: {myPars_bench.H_type_perm_weights}")
    print("myPars_no_cost")
    print(f"phi_H: {myPars_no_cost.phi_H}, wH_coeff: {myPars_no_cost.wH_coeff}, H_type_perm_weights: {myPars_no_cost.H_type_perm_weights}")
    print("myPars_no_wH")
    print(f"phi_H: {myPars_no_wH.phi_H}, wH_coeff: {myPars_no_wH.wH_coeff}, H_type_perm_weights: {myPars_no_wH.H_type_perm_weights}")
    print("myPars_all_high_H_types")
    print(f"phi_H: {myPars_all_high_H_types.phi_H}, wH_coeff: {myPars_all_high_H_types.wH_coeff}, H_type_perm_weights: {myPars_all_high_H_types.H_type_perm_weights}")
    print("myPars_no_MH")
    print(f"phi_H: {myPars_no_MH.phi_H}, wH_coeff: {myPars_no_MH.wH_coeff}, H_type_perm_weights: {myPars_no_MH.H_type_perm_weights}")



    quietly = True
    y_label_root = "Log Earnings"
    y_axis_label_var = "Variance" + " " + y_label_root
    my_label_lists = [["Benchmark"], ["No time cost"], ["No wH"], ["All high type"],["No mental health"]]
    out_path = main_path + "trans_output_test/"

    fig_var_benchmark, ax_var_benchmark = plot_ineq.plot_var_log_sim(myPars_bench,sims_bench["lab_earnings"], y_axis_label_var, out_path, quietly)
    fig_var_no_cost, ax_var_no_cost = plot_ineq.plot_var_log_sim(myPars_no_cost, sims_no_cost["lab_earnings"], y_axis_label_var, out_path, quietly)
    fig_var_no_wH, ax_var_no_wH = plot_ineq.plot_var_log_sim(myPars_no_wH, sims_no_wH["lab_earnings"], y_axis_label_var, out_path, quietly)
    fig_var_all_high_H_types, ax_var_all_high_H_types = plot_ineq.plot_var_log_sim(myPars_all_high_H_types, sims_all_high_H_types["lab_earnings"], y_axis_label_var, out_path, quietly)
    fig_var_no_MH, ax_var_no_MH = plot_ineq.plot_var_log_sim(myPars_no_MH, sims_no_MH["lab_earnings"], y_axis_label_var, out_path, quietly)

    bench_perc_plot_list = plot_ineq.plot_many_sim_perc_ratio(myPars_bench, np.log(np.where(sims_bench["lab_earnings"]>0,sims_bench["lab_earnings"],1e-3)), y_label_root, out_path, quietly)
    no_cost_perc_plot_list = plot_ineq.plot_many_sim_perc_ratio(myPars_no_cost, np.log(np.where(sims_no_cost["lab_earnings"]>0,sims_no_cost["lab_earnings"],1e-3)), y_label_root, out_path, quietly)
    no_wH_perc_plot_list = plot_ineq.plot_many_sim_perc_ratio(myPars_no_wH, np.log(np.where(sims_no_wH["lab_earnings"]>0,sims_no_wH["lab_earnings"],1e-3)), y_label_root, out_path, quietly)
    all_high_H_types_perc_plot_list = plot_ineq.plot_many_sim_perc_ratio(myPars_all_high_H_types, np.log(np.where(sims_all_high_H_types["lab_earnings"]>0,sims_all_high_H_types["lab_earnings"],1e-3)), y_label_root, out_path, quietly)
    no_MH_perc_plot_list = plot_ineq.plot_many_sim_perc_ratio(myPars_no_MH, np.log(np.where(sims_no_MH["lab_earnings"]>0,sims_no_MH["lab_earnings"],1e-3)), y_label_root, out_path, quietly)

    combined_plot_list = []
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    plot_save_path = main_path + "trans_output_test/" + "my_combined_plot" + ".pdf"
    fig, ax = tb.combine_plots([(fig_var_benchmark, ax_var_benchmark), 
                    (fig_var_no_cost, ax_var_no_cost), (fig_var_no_wH, ax_var_no_wH), 
                    (fig_var_all_high_H_types, ax_var_all_high_H_types), (fig_var_no_MH, ax_var_no_MH)], 
                        save_path=plot_save_path, label_lists = my_label_lists, quietly = True)


    # fig.savefig(plot_save_path)
    # fig.show()
    # plt.show()
    # tb.save_plot(combined_plot_list[0][0], plot_save_path)


    # for i in range(len(bench_perc_plot_list)):
    #     combined_plot_list.append(tb.combine_plots([bench_perc_plot_list[i], 
    #                     no_cost_perc_plot_list[i], no_wH_perc_plot_list[i], 
    #                     all_high_H_types_perc_plot_list[i], no_MH_perc_plot_list[i]], 
    #                     label_lists = my_label_lists, quietly = True))
    # plot_save_name = f"my_combined_plot{i}.pdf"
    # plot_save_path = out_path + "my_combined_plot" + ".pdf"
    # my_combined_plot = combined_plot_list[0][0]
    # my_combined_plot.savefig(plot_save_path)
    # Now switch to an interactive backend for displaying
    # plt.show()  # Show the plot
    # for plot_pair in combined_plot_list:
    #     plot_save_path = out_path + "my_combined_plot" + ".pdf"
    #     tb.save_plot(plot_pair[0], plot_save_path)


    plt.close('all')