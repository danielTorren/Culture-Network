"""Run multiple simulations varying two parameters
A module that use input data to generate data from multiple social networks varying two properties
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs. Useful for comparing the influences of these parameters
on each other to generate phase diagrams.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

#imports
from logging import raiseExceptions
from run import parallel_run, average_seed_parallel_run_mean_coefficient_variance
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap,  Normalize
from matplotlib.cm import get_cmap
import os
from plot import (
    live_print_culture_timeseries,
    print_culture_timeseries_vary_array,
    live_print_culture_timeseries_vary,
    live_compare_plot_animate_behaviour_scatter,
    live_phase_diagram_k_means_vary,
    print_culture_time_series_clusters_two_properties,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    live_print_heterogenous_culture_momentum_double,
    live_average_multirun_double_phase_diagram_mean,
    live_average_multirun_double_phase_diagram_C_of_V,
)
from utility import (
    produce_param_list_double,
    generate_title_list,
    createFolderSA,
    produceName_multi_run_n,
    produce_param_list_n,
    multi_n_save_variable_parameters_dict_list,
    multi_n_load_variable_parameters_dict_list,
    generate_vals_variable_parameters_and_norms,
    multi_n_save_combined_data,
    multi_n_load_combined_data,
    produce_param_list_n_double,
    multi_n_load_mean_data_list,
    multi_n_load_coefficient_variance_data_list,
    multi_n_save_mean_data_list,
    multi_n_save_coefficient_variance_data_list,
)

#constants
params = {
    "total_time": 30,
    "delta_t": 0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.1,
    "phi_upper": 1.0,
    "N": 100,
    "M": 3,
    "K": 20,
    "prob_rewire": 0.05,
    "set_seed": 1,
    "culture_momentum_real": 5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.6,
    "present_discount_factor": 0.8,
    "inverse_homophily": 0.2,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1,
    "confirmation_bias": 20,
}

params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

#behaviours!
if params["harsh_data"]:#trying to create a polarised society!
    params["green_extreme_max"]= 8
    params["green_extreme_min"]= 2
    params["green_extreme_prop"]= 2/5
    params["indifferent_max"]= 2
    params["indifferent_min"]= 2
    params["indifferent_prop"]= 1/5
    params["brown_extreme_min"]= 2
    params["brown_extreme_max"]= 8
    params["brown_extreme_prop"]= 2/5
    if params["green_extreme_prop"] + params["indifferent_prop"] + params["brown_extreme_prop"] != 1:
        raise Exception("Invalid proportions")
else:
    params["alpha_attitude"] = 0.2
    params["beta_attitude"] = 0.2
    params["alpha_threshold"] = 1
    params["beta_threshold"] = 1

###PLOT STUFF
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1)

#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)

#log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

alpha_quick, alpha_normal, alpha_lagard = 0.9,0.7,0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"

#print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1

min_val = 1e-3

dpi_save = 1200

min_k,max_k = 2,10#N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

SINGLE = 0

if __name__ == "__main__":
    nrows = 4
    ncols = 4#due to screen ratio want more cols than rows usually
    reps = nrows*ncols

    """Runs parallel loops over reps of each variable parameters dict entry"""
    variable_parameters_dict = {
        #"discount_factor": {"property":"discount_factor","min":-1, "max":0 , "title": r"$\delta$", "divisions": "log", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o"}, 
        "inverse_homophily": {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "v", "reps": nrows}, 
        "confirmation_bias": {"property":"confirmation_bias","min":0, "max":30 , "title": r"$\theta$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p", "reps": ncols}, 
        #"prob_rewire": {"property":"prob_rewire","min":0.0, "max":1 , "title": r"$p_r$", "divisions": "linear", "cmap": get_cmap("Purples"), "cbar_loc": "right", "marker": "d"}, 
        #"learning_error_scale": {"property":"learning_error_scale","min":0.0,"max":1.0 , "title": r"$\epsilon$", "divisions": "linear", "cmap": get_cmap("Oranges"), "cbar_loc": "right", "marker": "*"},
        #"N": {"property": "N","min":50,"max":200, "title": r"$N$", "divisions": "linear", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o"}, 
        #"M": {"property":"M","min":1,"max": 10, "title": r"$M$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p"}, 
        #{"property":"K","min":2,"max":30 , "title": r"$K$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "p"}, 
    }

    param_row = "inverse_homophily"
    param_col = "confirmation_bias"

    reps_row = variable_parameters_dict[param_row]["reps"]
    reps_col = variable_parameters_dict[param_col]["reps"]

    reps = reps_row*reps_col #64#32 # make multiple of cpu core number for efficiency

    param_min_row = variable_parameters_dict[param_row]["min"]
    param_max_row = variable_parameters_dict[param_row]["max"]
    param_min_col = variable_parameters_dict[param_col]["min"]
    param_max_col = variable_parameters_dict[param_col]["max"]

    property_row = variable_parameters_dict[param_row]["title"]
    property_col = variable_parameters_dict[param_col]["title"] 


    if SINGLE:

        fileName = "results/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (param_col,param_row,str(params["N"]),str(params["time_steps_max"]),str(params["K"]),str(param_min_col), str(param_max_col), str(param_min_row), str(param_max_row), str(reps))
        print("fileName: ", fileName)
        createFolderSA(fileName)

        ### GENERATE DATA
        #print(np.linspace(param_min_row,param_max_row, nrows), type(np.linspace(param_min_row,param_max_row, nrows)))
        #print(np.asarray([0.05, 0.5, 1.0, 5.0]), type(np.asarray([0.05, 0.5, 1.0, 5.0])))
        #quit()
        row_list = np.linspace(param_min_row,param_max_row, nrows)#np.asarray([0.08, 0.5, 1.0, 5.0])
        col_list = np.linspace(param_min_col,param_max_col, ncols)#np.asarray([0.08, 0.5, 1.0, 5.0])#
        params_list = produce_param_list_double(params,param_col,col_list,param_row,row_list)
        
        data_list  = parallel_run(params_list)  
        data_array = np.reshape(data_list, (len(row_list), len(col_list)))
        title_list = generate_title_list(property_col,col_list,property_row,row_list, round_dec)


        ### PLOTS
        live_print_culture_timeseries_vary(fileName, data_list, param_row, param_col,title_list, nrows, ncols,  dpi_save)
        #print_culture_timeseries_vary_array(fileName, data_array, property_col, col_list,property_row,row_list,  nrows, ncols , dpi_save)
        #live_phase_diagram_k_means_vary(fileName, data_array, property_row,  row_list,property_col,col_list,min_k,max_k,size_points, cmap_weighting,dpi_save)
        #print_culture_time_series_clusters_two_properties(fileName,data_array, row_list, col_list,property_row, property_col, min_k,max_k,size_points, alpha_val, min_culture_distance,"DTW", nrows, ncols, dpi_save, round_dec)

        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,property_col,col_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,property_col,col_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, nrows, ncols,property_col,col_list)

    else: 
        """Runs parallel loops over reps of each variable parameters dict entry"""
        RUN = 0

        #AVERAGE OVER MULTIPLE RUNS
        seed_list = [1,2,3]# [1,2,3,4,5] ie 5 reps per run!
        params["seed_list"] = seed_list
        average_reps = len(seed_list)


        variable_parameters_dict = generate_vals_variable_parameters_and_norms(variable_parameters_dict)

        if RUN:

            property_varied_values_row = variable_parameters_dict[param_row]["vals"]
            property_varied_values_col = variable_parameters_dict[param_col]["vals"]
            

            fileName = "results/average_%s_%s_%s_%s_%s_%s_%s" % (param_col,param_row,str(params["N"]),str(params["time_steps_max"]),str(params["K"]),str(reps), str(average_reps))

            print("fileName: ", fileName)
            createFolderSA(fileName)
            
            ### GENERATE PARAMS 
            
            params_list = produce_param_list_n_double(params,variable_parameters_dict, param_row,param_col)

            print(reps_row,reps_col,len(params_list ))

            ### GENERATE DATA
            results_mean, results_coefficient_variance = average_seed_parallel_run_mean_coefficient_variance(params_list) 
            print("results_mean",results_mean)

            #save the data and params_list
            multi_n_save_variable_parameters_dict_list(variable_parameters_dict, fileName)
            multi_n_save_mean_data_list(results_mean, fileName)
            multi_n_save_coefficient_variance_data_list(results_coefficient_variance,fileName)

            matrix_mean = results_mean.reshape((reps_row, reps_col))
            matrix_coefficient_variance = results_coefficient_variance.reshape((reps_row, reps_col))

        else:
            fileName = "results/average_confirmation_bias_inverse_homophily_100_1000_20_256_5"

            if os.path.exists(fileName + '/variable_parameters_dict.pkl'): 
                variable_parameters_dict = multi_n_load_variable_parameters_dict_list(fileName)

            param_row = "inverse_homophily"
            param_col = "confirmation_bias"

            reps_row = variable_parameters_dict[param_row]["reps"]
            reps_col = variable_parameters_dict[param_col]["reps"]

            property_row = variable_parameters_dict[param_row]["title"]
            property_col = variable_parameters_dict[param_col]["title"] 

            property_varied_values_row = variable_parameters_dict[param_row]["vals"]
            property_varied_values_col = variable_parameters_dict[param_col]["vals"]

            if os.path.exists(fileName + '/mean_data_list.pkl'):
                results_mean = multi_n_load_mean_data_list(fileName)
            else:
                raiseExceptions("results mean missing!")

            if os.path.exists(fileName + '/coefficient_variance_data_list.pkl'):
                results_coefficient_variance = multi_n_load_coefficient_variance_data_list(fileName)
            else:
                raiseExceptions("coefficient variance missing!")

            matrix_mean = results_mean.reshape((reps_row, reps_col))
            matrix_coefficient_variance = results_coefficient_variance.reshape((reps_row, reps_col))


        live_average_multirun_double_phase_diagram_mean(fileName, matrix_mean, property_row, property_varied_values_row,property_col,property_varied_values_col, get_cmap("Blues"),dpi_save,round_dec)
        live_average_multirun_double_phase_diagram_C_of_V(fileName, matrix_coefficient_variance, property_row, property_varied_values_row,property_col,property_varied_values_col, get_cmap("Reds"),dpi_save,round_dec)

    plt.show()