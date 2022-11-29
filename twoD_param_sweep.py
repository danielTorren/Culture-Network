"""Run multiple simulations varying two parameters
A module that use input data to generate data from multiple social networks varying two properties
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs. Useful for comparing the influences of these parameters
on each other to generate phase diagrams.

TWO MODES 
    The two parameters can be varied covering a 2D plane of points. This can either be done in SINGLE = True where individual 
    runs are used as the output and gives much greater variety of possible plots but all these plots use the same initial
    seed. Alternatively can be run such that multiple averages of the simulation are produced and then the data accessible 
    is the emissions, mean identity, variance of identity and the coefficient of variance of identity.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import json
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
from resources.plot import (
    live_print_culture_timeseries,
    print_culture_timeseries_vary_array,
    live_print_culture_timeseries_vary,
    live_compare_plot_animate_behaviour_scatter,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    # live_print_heterogenous_culture_momentum_double,
    live_average_multirun_double_phase_diagram_mean,
    live_average_multirun_double_phase_diagram_mean_alt,
    live_average_multirun_double_phase_diagram_C_of_V_alt,
    live_average_multirun_double_phase_diagram_C_of_V,
    double_phase_diagram,
    double_phase_diagram_using_meanandvariance,
    double_matrix_plot,
    multi_line_matrix_plot,
    multi_line_matrix_plot_divide_through,
    double_matrix_plot_ab,
    double_matrix_plot_cluster
)


from resources.utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
    save_object,
    load_object
)
from resources.multi_run_2D_param import (
    generate_title_list,
    shot_two_dimensional_param_run,
    load_data_shot,
    av_two_dimensional_param_run,
    load_data_av,
    reshape_results_matricies,
    produce_param_list_n_double,
)
from resources.run import cluster_data_run
import numpy as np

# run bools
RUN = 1 # run or load in previously saved data
SINGLE = 0 # determine if you runs single shots or study the averages over multiple runs for each experiment
cluster_count_run = 1
ab_plot = 0
plot_conf_attiude = 0
plot_multi_line_divide = 1
plot_multi_line = 0

fileName = "results/twoD_Average_M_confirmation_bias_200_2000_20_10_402_5"
#"results/twoD_Average_confirmation_bias_M_200_3000_20_70_20_5"#"results/twoD_Average_confirmation_bias_a_attitude_200_3000_20_64_64_5"#"
#"results/twoD_Average_confirmation_bias_a_attitude_200_3000_20_64_64_5"
#"results/twoD_Average_action_observation_I_a_attitude_200_2000_20_64_64_5"
#twoD_Average_M_confirmation_bias_200_2000_20_40_64_5
#twoD_Average_homophily_confirmation_bias_200_2000_20_64_64_5
#twoD_Average_M_confirmation_bias_200_2000_20_10_402_5


###PLOT STUFF
dpi_save = 1200
round_dec = 2
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
node_size = 50
cmap = LinearSegmentedColormap.from_list(
    "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
)
norm_zero_one = Normalize(vmin=0, vmax=1)

"""
{
    "row":{"property":"M","min":1, "max": 20, "title": "M","divisions": "linear", "reps": 20}, 
    "col":{"property":"confirmation_bias","min":0, "max":90 , "title": "Confirmation bias, \\theta","divisions": "linear", "reps": 50}  
}

"""

if __name__ == "__main__":
    if SINGLE:
        if RUN:
            # load base params
            f_base_params = open("constants/base_params.json")
            params = json.load(f_base_params)
            f_base_params.close()
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

            # load variable params
            f_variable_parameters = open(
                "constants/variable_parameters_dict_2D.json"
            )
            variable_parameters_dict = json.load(f_variable_parameters)
            f_variable_parameters.close()

            variable_parameters_dict = generate_vals_variable_parameters_and_norms(
                variable_parameters_dict
            )

            fileName = "results/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                variable_parameters_dict["col"]["property"],
                variable_parameters_dict["row"]["property"],
                str(params["N"]),
                str(params["time_steps_max"]),
                str(params["K"]),
                str(variable_parameters_dict["col"]["min"]),
                str(variable_parameters_dict["col"]["max"]),
                str(variable_parameters_dict["row"]["min"]),
                str(variable_parameters_dict["row"]["max"]),
                variable_parameters_dict["col"]["reps"],
                variable_parameters_dict["row"]["reps"],
            )
            print("fileName: ", fileName)

            title_list = generate_title_list(
                variable_parameters_dict["col"]["title"],
                variable_parameters_dict["col"]["vals"],
                variable_parameters_dict["row"]["title"],
                variable_parameters_dict["row"]["vals"],
                round_dec,
            )
            data_list, data_array = shot_two_dimensional_param_run(
                fileName,
                params,
                variable_parameters_dict,
                variable_parameters_dict["row"]["reps"],
                variable_parameters_dict["col"]["reps"],
            )

        else:
            variable_parameters_dict, data_list, data_array = load_data_shot(fileName)
            """
            title_list = generate_title_list(
                variable_parameters_dict["col"]["title"],
                variable_parameters_dict["col"]["vals"],
                variable_parameters_dict["row"]["title"],
                variable_parameters_dict["row"]["vals"],
                round_dec,
            )
            """

            title_list = generate_title_list(
                "AO_S",
                variable_parameters_dict["col"]["vals"],
                "AO_I",
                variable_parameters_dict["row"]["vals"],
                round_dec,
            )


        ### PLOTS FOR SINGLE SHOT RUNS
        live_print_culture_timeseries_vary(
            fileName,
            data_list,
            variable_parameters_dict["row"]["property"],
            variable_parameters_dict["col"]["property"],
            title_list,
            variable_parameters_dict["row"]["reps"],
            variable_parameters_dict["col"]["reps"],
            dpi_save,
        )
        # BROKEN print_culture_timeseries_vary_array(fileName, data_array, param_col,property_col,property_varied_values_col,param_row, property_row,property_varied_values_row,  reps_row, reps_col , dpi_save)

        # ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, reps_col, reps_row,property_col,property_varied_values_col)
        # ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, reps_row, reps_col,property_col,property_varied_values_col)
        # ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, reps_row, reps_col,property_col,property_varied_values_col)

    else:
        if RUN:
            # load base params
            f_base_params = open("constants/base_params.json")
            params = json.load(f_base_params)
            f_base_params.close()
            params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

            # load variable params
            f_variable_parameters = open(
                "constants/variable_parameters_dict_2D.json"
            )
            variable_parameters_dict = json.load(f_variable_parameters)
            f_variable_parameters.close()

            # AVERAGE OVER MULTIPLE RUNS
            variable_parameters_dict = generate_vals_variable_parameters_and_norms(
                variable_parameters_dict
            )

            fileName = "results/twoD_Average_%s_%s_%s_%s_%s_%s_%s_%s" % (
                variable_parameters_dict["col"]["property"],
                variable_parameters_dict["row"]["property"],
                str(params["N"]),
                str(params["time_steps_max"]),
                str(params["K"]),
                str(variable_parameters_dict["col"]["reps"]),
                str(variable_parameters_dict["row"]["reps"]),
                len(params["seed_list"]),
            )
            print("fileName: ", fileName)

            if cluster_count_run:
                
                    createFolder(fileName)
                    s = np.linspace(0,1,50)
                    params_list = produce_param_list_n_double(params, variable_parameters_dict)
                    (
                        results_emissions,
                        results_mu,
                        results_var,
                        results_coefficient_of_variance,
                        results_clusters_count,
                    ) = cluster_data_run(params_list,s)

                    # save the data and params_list
                    save_object(
                        variable_parameters_dict, fileName + "/Data", "variable_parameters_dict"
                    )
                    save_object(results_emissions, fileName + "/Data", "results_emissions")
                    save_object(results_mu, fileName + "/Data", "results_mu")
                    save_object(results_var, fileName + "/Data", "results_var")
                    save_object(results_coefficient_of_variance,fileName + "/Data","results_coefficient_of_variance")
                    save_object(results_clusters_count,fileName + "/Data","results_clusters_count")
            else:
                (
                    results_emissions,
                    results_mu,
                    results_var,
                    results_coefficient_of_variance,
                ) = av_two_dimensional_param_run(fileName, variable_parameters_dict, params)


            
        else:
            createFolder(fileName)

            if cluster_count_run:
                variable_parameters_dict = load_object(
                    fileName + "/Data", "variable_parameters_dict"
                )
                results_emissions = load_object(fileName + "/Data", "results_emissions")
                results_mu = load_object(fileName + "/Data", "results_mu")
                results_var = load_object(fileName + "/Data", "results_var")
                results_coefficient_of_variance = load_object(
                    fileName + "/Data", "results_coefficient_of_variance"
                )

                results_clusters_count = load_object(fileName + "/Data","results_clusters_count")

                reps_row, reps_col = variable_parameters_dict["row"]["reps"],variable_parameters_dict["col"]["reps"]
                matrix_emissions = results_emissions.reshape((reps_row, reps_col))
                matrix_mu = results_mu.reshape((reps_row, reps_col))
                matrix_var = results_var.reshape((reps_row, reps_col))
                matrix_coefficient_of_variance = results_coefficient_of_variance.reshape((reps_row, reps_col))
                matrix_clusters_count = results_clusters_count.reshape((reps_row, reps_col))
            else:    
                (
                    variable_parameters_dict,
                    results_emissions,
                    results_mu,
                    results_var,
                    results_coefficient_of_variance,
                ) = load_data_av(fileName)

                ###PLOTS FOR STOCHASTICALLY AVERAGED RUNS
                (
                    matrix_emissions,
                    matrix_mu,
                    matrix_var,
                    matrix_coefficient_of_variance,
                ) = reshape_results_matricies(
                    results_emissions,
                    results_mu,
                    results_var,
                    results_coefficient_of_variance,
                    variable_parameters_dict["row"]["reps"],
                    variable_parameters_dict["col"]["reps"],
            )

        #double_phase_diagram(fileName, matrix_emissions, r"Total normalised emissions $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save)
        #double_phase_diagram(fileName, matrix_mu, r"Average identity, $\mu$", "mu",variable_parameters_dict, get_cmap("Blues"),dpi_save)
        #double_phase_diagram(fileName, matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Greens"),dpi_save)
        #double_phase_diagram(fileName, matrix_coefficient_of_variance, r"Identity coefficient of variance, $\sigma/\mu$", "coefficient_of_variance",variable_parameters_dict, get_cmap("Oranges"),dpi_save)

        if ab_plot:
            ############################
            #PLOT THE AB MATRIX TO SHOW CORRESPONDANCE WITH OUTPUT

            #col_ticks_label = [0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            #col_ticks_pos = [0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_x_matrix*((x - min_x_val)/(max_x_val- min_x_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            #print("x_ticks_pos",x_ticks_pos)

                    
            #row_ticks_label = [0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            #row_ticks_pos = [0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_y_matrix*((y - min_y_val)/(max_y_val- min_y_val)))) for y in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            
            #print("row",row_ticks_pos,row_ticks_label)
            #print("col",col_ticks_pos,col_ticks_label)

            #col_ticks_pos = [0, 6, 64] 
            #col_ticks_label = [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"] 
            #row_ticks_pos = [0, 6, 64] 
            #row_ticks_label = [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"]

            double_matrix_plot_ab(fileName, matrix_emissions, r"Total normalised emissions $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label)
            #double_matrix_plot(fileName, matrix_mu, r"Average identity, $\mu$", "mu",variable_parameters_dict, get_cmap("Blues"),dpi_save)
            #print("variable_parameters_dict",variable_parameters_dict)
            #double_phase_diagram(fileName, matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Oranges"),dpi_save)
            #double_matrix_plot_ab(fileName,matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Oranges"),dpi_save)
            #double_matrix_plot(fileName,matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Oranges"),dpi_save,x_ticks_pos,y_ticks_pos,x_ticks_label,y_ticks_label)
            #double_matrix_plot(fileName, matrix_coefficient_of_variance, r"Identity coefficient of variance, $\sigma/\mu$", "coefficient_of_variance",variable_parameters_dict, get_cmap("Oranges"),dpi_save)


        #when using discrete or interger variables ie, K,N,M
        if plot_conf_attiude:
            col_dict = variable_parameters_dict["col"]
            row_dict = variable_parameters_dict["row"]
            #### FOR confimation bias vs attitude polarisation
            index_len_x_matrix = 63
            max_x_val = 90
            min_x_val = -10
            col_ticks_label = [-10,0,10,20,30,40,50,60,70,80,90]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            col_ticks_pos = [-10,0,10,20,30,40,50,60,70,80,90]#[int(round(index_len_x_matrix*((x - min_x_val)/(max_x_val- min_x_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            #print("x_ticks_pos",x_ticks_pos)

                    
            index_len_y_matrix = 63
            max_y_val = 0.05
            min_y_val =  2.0
            row_ticks_label = [0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            row_ticks_pos = [0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_y_matrix*((y - min_y_val)/(max_y_val- min_y_val)))) for y in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            
            print("row",row_ticks_pos,row_ticks_label)
            print("col",col_ticks_pos,col_ticks_label)

            row_label = r"Attitude Beta parameters, $(a,b)$"#r"Number of behaviours per agent, M"
            col_label = r'Confirmation bias, $\theta$'
            y_label = r"Identity variance, $\sigma^2$"

            multi_line_matrix_plot(fileName,matrix_var, col_dict["vals"], row_dict["vals"],"variance", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
            multi_line_matrix_plot(fileName,matrix_var, col_dict["vals"], row_dict["vals"],"variance", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label


        if plot_multi_line:
            col_dict = variable_parameters_dict["col"]
            row_dict = variable_parameters_dict["row"]

            #print("col",col_dict["property"])

            select_val_col = 5
            select_val_row = 2

            #quit()
            #print("col",y_ticks_pos,y_ticks_label)

            col_axis_x = 0
            
            #multi_line_matrix_plot(fileName,matrix_var, col_dict["vals"], row_dict["vals"],"variance", get_cmap("Blues"),dpi_save,x_ticks_pos, x_ticks_label, y_ticks_pos, y_ticks_label,col_axis_x)#y_ticks_pos, y_ticks_label
            
            #######REMOVING THE NEGATIVE VALS FROM THAT ONE PARTICULAR RUN
            Z = matrix_var[::,10::]
            col_vals = col_dict["vals"][10:]
            row_vals = row_dict["vals"]


            col_ticks_pos = [col_vals[y] for y in range(len(col_vals))  if y % select_val_row == 0]
            col_ticks_label = [col_vals[y] for y in range(len(col_vals))  if y % select_val_row == 0]
        
            row_ticks_pos  = [row_vals[y] for y in range(len(row_vals))  if y % select_val_row == 0]#[y for y in range(len(row_dict["vals"]))  if y % select_val_row == 0]
            row_ticks_label  = [row_vals[y] for y in range(len(row_vals))  if y % select_val_row == 0]

            print("row",row_ticks_pos,row_ticks_label)

            multi_line_matrix_plot(fileName,Z, col_vals, row_dict["vals"],"variance", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x)#y_ticks_pos, y_ticks_label
        

        if plot_multi_line_divide:
            col_dict = variable_parameters_dict["col"]#confirmation bias
            row_dict = variable_parameters_dict["row"]#m

            #print("col",col_dict["property"])

            select_val_col = 10
            select_val_row = 2

            #print("row",row_ticks_pos,row_ticks_label)
            #quit()
            #print("col",y_ticks_pos,y_ticks_label)

            #col_axis_x = 0
            
            #multi_line_matrix_plot(fileName,matrix_var, col_dict["vals"], row_dict["vals"],"variance", get_cmap("Blues"),dpi_save,x_ticks_pos, x_ticks_label, y_ticks_pos, y_ticks_label,col_axis_x)#y_ticks_pos, y_ticks_label
            
            #######REMOVING THE NEGATIVE VALS FROM THAT ONE PARTICULAR RUN
            Z = matrix_var[::,10::]
            col_vals = col_dict["vals"][10:]

            #######Divde through by first value

            #print("matrix_var[0]",Z[0])
            #print("shape matrix", Z.shape)
            normalized_m = Z/Z[0]
            #print("normalized_m",normalized_m.shape)
            matrix_norm_var_edit = normalized_m[1:]
            #print()
            row_vals = row_dict["vals"][1:]

            round_dec = 2

            index_len_col_matrix = len(col_vals)
            max_col_val = 58.98550724637681
            min_col_val = 0.1449275362318847
            
            col_ticks_label = [0,10,20,30,40,50,60]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
            col_ticks_pos = [0,10,20,30,40,50,60]#[int(round(index_len_col_matrix*((x - min_col_val)/(max_col_val- min_col_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        
            row_ticks_pos  = [row_vals[y] for y in range(len(row_vals))  if y % select_val_row == 0]#[y for y in range(len(row_dict["vals"]))  if y % select_val_row == 0]
            row_ticks_label  = [row_vals[y] for y in range(len(row_vals))  if y % select_val_row == 0]

            print("row",row_ticks_pos,row_ticks_label)
            print("col",col_ticks_pos,col_ticks_label)

            #multi_line_matrix_plot_divide_through(fileName,matrix_norm_var_edit, col_vals, row_vals,"variance", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,0)#y_ticks_pos, y_ticks_label
            multi_line_matrix_plot_divide_through(fileName,matrix_norm_var_edit, col_vals, row_vals,"variance", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,1)#y_ticks_pos, y_ticks_label

        if cluster_count_run:
            double_matrix_plot_cluster(fileName,matrix_clusters_count,variable_parameters_dict, get_cmap("Purples"),dpi_save)

        #only for the a or b beta parameters
        #double_phase_diagram_using_meanandvariance(fileName, matrix_emissions, r"Total normalised emissions, $E/NM$", "emissions",variable_parameters_dict, get_cmap("Reds"),dpi_save)
        #double_phase_diagram_using_meanandvariance(fileName,matrix_mu,r"Average identity, $\mu$","mu",variable_parameters_dict,get_cmap("Blues"),dpi_save,)
        #double_phase_diagram_using_meanandvariance(fileName, matrix_var, r"Identity variance, $\sigma^2$", "variance",variable_parameters_dict, get_cmap("Greens"),dpi_save)
        #double_phase_diagram_using_meanandvariance(fileName,matrix_coefficient_of_variance,r"Identity coefficient of variance, $\sigma/\mu$","coefficient_of_variance",variable_parameters_dict,get_cmap("Oranges"),dpi_save,)

    plt.show()
