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
    double_matrix_plot_cluster,
    double_matrix_plot_cluster_ratio,
    double_matrix_plot_cluster_multi,
    double_matrix_plot_cluster_var_multi,
    double_contour_plot_cluster,
)
from resources.utility import (
    createFolder,
    generate_vals_variable_parameters_and_norms,
    save_object,
    load_object,
    produce_name_datetime,
    calc_num_clusters_set_bandwidth,
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
from resources.run import (
    single_stochstic_emissions_run
)
import numpy as np

# run bools
RUN = 0 # run or load in previously saved data
SINGLE = 1 # determine if you runs single shots or study the averages over multiple runs for each experiment


fileName = "results/splitting_eco_warriors_single_14_47_07__03_01_2023"
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

norm_zero_one = Normalize(vmin=0, vmax=1)

if __name__ == "__main__":
    if SINGLE:
        if RUN:
            # load base params
            f_base_params = open("constants/base_params.json")
            base_params = json.load(f_base_params)
            f_base_params.close()
            base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

            # load variable params
            variable_parameters_dict = {
                "col":{"property":"confirmation_bias","min":0, "max":100 , "title": "Confirmation bias, $\\theta$","divisions": "linear", "reps": 100},  
                "row":{"property":"green_N","min":0, "max": 100, "title": "Eco warrior count","divisions": "linear", "reps": 100}, 
            }

            variable_parameters_dict = generate_vals_variable_parameters_and_norms(
                variable_parameters_dict
            )

            root = "splitting_eco_warriors_single"
            fileName = produce_name_datetime(root)
            print("fileName:", fileName)
            #print("fileName: ", fileName)

            params_dict_list = produce_param_list_n_double(base_params, variable_parameters_dict)

            emissions_list = single_stochstic_emissions_run(params_dict_list)

            matrix_emissions = emissions_list.reshape((variable_parameters_dict["row"]["reps"], variable_parameters_dict["col"]["reps"]))

            createFolder(fileName)
    
            save_object(base_params, fileName + "/Data", "base_params")
            save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
            save_object(matrix_emissions, fileName + "/Data", "matrix_emissions")

        else:
            base_params = load_object(fileName + "/Data", "base_params")
            variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
            matrix_emissions = load_object(fileName + "/Data", "matrix_emissions")

        col_dict = variable_parameters_dict["col"]
        row_dict = variable_parameters_dict["row"]

        #### FOR confimation bias vs attitude polarisation
        index_len_x_matrix = col_dict["reps"]
        max_x_val = col_dict["max"]
        min_x_val = col_dict["min"]
        col_ticks_label = np.arange(min_x_val, min_x_val, 10)#[-10,0,10,20,30,40,50,60,70,80,90]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        col_ticks_pos = np.arange(min_x_val, min_x_val, 10)#[-10,0,10,20,30,40,50,60,70,80,90]#[int(round(index_len_x_matrix*((x - min_x_val)/(max_x_val- min_x_val)))) for x in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        #print("x_ticks_pos",x_ticks_pos)

                
        index_len_y_matrix =row_dict["reps"]
        max_y_val = row_dict["max"]
        min_y_val = row_dict["min"]
        row_ticks_label = np.arange(min_y_val, min_y_val, 10)#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[-10,0,10,20,30,40,50,60]#[col_dict["vals"][x] for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        row_ticks_pos = np.arange(min_y_val, min_y_val, 10)#[0.05,0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00]#[int(round(index_len_y_matrix*((y - min_y_val)/(max_y_val- min_y_val)))) for y in col_ticks_label]#[0,30,70,50]#[0,10,20,30,40,50,60,70]#[x for x in range(len(col_dict["vals"]))  if x % select_val_x == 0]
        
        #print("row",row_ticks_pos,row_ticks_label)
        #print("col",col_ticks_pos,col_ticks_label)

        row_label = r"Eco-warriors count"#r"Number of behaviours per agent, M"
        col_label = r'Confirmation bias, $\theta$'#r'Confirmation bias, $\theta$'
        y_label = r"Final emissions, $E$"#r"Identity variance, $\sigma^2$"

        multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 0, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        multi_line_matrix_plot(fileName,matrix_emissions, col_dict["vals"], row_dict["vals"],"emissions", get_cmap("plasma"),dpi_save,col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label, 1, col_label, row_label, y_label)#y_ticks_pos, y_ticks_label
        #multi_line_matrix_plot(fileName, Z, col_vals, row_vals,  Y_param, cmap, dpi_save, col_ticks_pos, col_ticks_label, row_ticks_pos, row_ticks_label,col_axis_x, col_label, row_label, y_label)
        
        #### two D plot of emissions with confimation bias and number of eco warriors


    plt.show()
