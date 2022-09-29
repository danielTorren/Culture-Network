from logging import raiseExceptions
from run import average_seed_parallel_run_mean_coefficient_variance_multi_run_n
import matplotlib.pyplot as plt
import numpy as np
import os
from utility import (
    createFolderSA,
    produceName_multi_run_n,
    produce_param_list_n,
    multi_n_save_variable_parameters_dict_list,
    multi_n_load_variable_parameters_dict_list,
    generate_vals_variable_parameters_and_norms,
    multi_n_save_combined_data,
    multi_n_load_combined_data,
)
from matplotlib.cm import get_cmap
from plot import (
    live_average_multirun_n_diagram_mean_coefficient_variance,
    live_average_multirun_n_diagram_mean_coefficient_variance_cols
)

params = {
    "total_time": 30,
    "delta_t": 0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_list_lower": 0.1,
    "phi_list_upper": 1.0,
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
    params["alpha_attitude"] = 1
    params["beta_attitude"] = 1
    params["alpha_threshold"] = 1
    params["beta_threshold"] = 1

###PLOT STUFF
dpi_save = 1200

RUN = 1

if __name__ == "__main__":

    #reps = 64#64#32 # make multiple of cpu core number for efficiency

    if RUN:
        #AVERAGE OVER MULTIPLE RUNS
        seed_list = [1,2,3,4,5]#ie 5 reps per run!
        params["seed_list"] = seed_list
        average_reps = len(seed_list)

        reps = 262 #total reps 
        fileName = "results/multi_run_n_%s_%s_%s_%s_%s" % (str(params["N"]),str(params["time_steps_max"]),str(params["K"]), str(len(seed_list)), str(reps))

        # variable parameters

        """Runs parallel loops over reps of each variable parameters dict entry"""
        variable_parameters_dict = {
            "discount_factor": {"property":"discount_factor","min":-2, "max":0 , "title": r"$\delta$", "divisions": "log", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o", "reps": 16}, 
            "inverse_homophily": {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "v", "reps": 128}, 
            "confirmation_bias": {"property":"confirmation_bias","min":0, "max":40, "title": r"$\theta$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p", "reps":128}, 
            #"prob_rewire": {"property":"prob_rewire","min":0.0, "max":1 , "title": r"$p_r$", "divisions": "linear", "cmap": get_cmap("Purples"), "cbar_loc": "right", "marker": "d", "reps": 16}, 
            #"learning_error_scale": {"property":"learning_error_scale","min":0.0,"max":1.0 , "title": r"$\epsilon$", "divisions": "linear", "cmap": get_cmap("Oranges"), "cbar_loc": "right", "marker": "*", "reps": 16},
            #"N": {"property": "N","min":50,"max":200, "title": r"$N$", "divisions": "linear", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o", "reps": 8}, 
            #"M": {"property":"M","min":1,"max": 10, "title": r"$M$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p", "reps": 16}, 
            #{"property":"K","min":2,"max":30 , "title": r"$K$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "p", "reps": 16}, 
        }

        produceName_multi_run_n(variable_parameters_dict,fileName)
        
        print("fileName: ", fileName)
        createFolderSA(fileName)

        ### GENERATE PARAMS 
        variable_parameters_dict = generate_vals_variable_parameters_and_norms(variable_parameters_dict)
        params_list = produce_param_list_n(params,variable_parameters_dict)
        ### GENERATE DATA
        combined_data  = average_seed_parallel_run_mean_coefficient_variance_multi_run_n(params_list,variable_parameters_dict)  

        #save the data and params_list
        multi_n_save_variable_parameters_dict_list(variable_parameters_dict, fileName)
        multi_n_save_combined_data(combined_data, fileName)
    else:
        fileName = "results/multi_run_n_100_600_20_5_262"
        NEW_VARIABLES_PARAMS = 1

        if os.path.exists(fileName + '/variable_parameters_dict.pkl') and NEW_VARIABLES_PARAMS == 0: 
            variable_parameters_dict = multi_n_load_variable_parameters_dict_list(fileName)
        else:
            variable_parameters_dict = {
                "discount_factor": {"property":"discount_factor","min":-2, "max":0 , "title": r"$\delta$", "divisions": "log", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o", "reps": 16}, 
                "inverse_homophily": {"property":"inverse_homophily","min":0.0, "max": 1.0, "title": r"$h$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "v", "reps": 128}, 
                "confirmation_bias": {"property":"confirmation_bias","min":0, "max":40, "title": r"$\theta$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p", "reps":128}, 
                #"prob_rewire": {"property":"prob_rewire","min":0.0, "max":1 , "title": r"$p_r$", "divisions": "linear", "cmap": get_cmap("Purples"), "cbar_loc": "right", "marker": "d", "reps": 16}, 
                #"learning_error_scale": {"property":"learning_error_scale","min":0.0,"max":1.0 , "title": r"$\epsilon$", "divisions": "linear", "cmap": get_cmap("Oranges"), "cbar_loc": "right", "marker": "*", "reps": 16},
                #"N": {"property": "N","min":50,"max":200, "title": r"$N$", "divisions": "linear", "cmap": get_cmap("Reds"), "cbar_loc": "right", "marker": "o", "reps": 8}, 
                #"M": {"property":"M","min":1,"max": 10, "title": r"$M$", "divisions": "linear", "cmap": get_cmap("Greens"), "cbar_loc": "right", "marker": "p", "reps": 16}, 
                #{"property":"K","min":2,"max":30 , "title": r"$K$", "divisions": "linear", "cmap": get_cmap("Blues"), "cbar_loc": "right", "marker": "p", "reps": 16}, 
            }

            variable_parameters_dict = generate_vals_variable_parameters_and_norms(variable_parameters_dict)

        if os.path.exists(fileName + '/combined_data.pkl'):
            combined_data = multi_n_load_combined_data(fileName)
        else:
            raiseExceptions("combined data missing!")

    ### PLOTS

    #plot_a = live_average_multirun_n_diagram_mean_coefficient_variance(fileName, mean_data_list,coefficient_variance_data_list ,variable_parameters_dict,dpi_save)
    plot_b = live_average_multirun_n_diagram_mean_coefficient_variance(fileName, combined_data ,variable_parameters_dict,dpi_save,)
    plot_c = live_average_multirun_n_diagram_mean_coefficient_variance_cols(fileName, combined_data ,variable_parameters_dict,dpi_save)
    plt.show()