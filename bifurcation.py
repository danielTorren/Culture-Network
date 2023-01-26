"""Produce bifurcation plots

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.cm import get_cmap
from resources.utility import createFolder,produce_name_datetime,save_object,load_object,calc_pos_clusters_set_bandwidth, get_cluster_list,generate_vals_variable_parameters_and_norms
from resources.run import parallel_run, parallel_run_sa,culture_data_run,one_seed_culture_data_run
from resources.plot import (
    bifurcation_plot,
    bifurcation_plot_stochastic,
    bifurcation_plot_one_seed_two_params,
    bifurcation_heat_map_stochastic,
    bifurcation_plot_culture_or_not,
)
from resources.multi_run_single_param import (
    produce_param_list,
)
from resources.multi_run_2D_param import (
    produce_param_list_n_double,
)

# constants
###PLOT STUFF
node_size = 100
cmap = LinearSegmentedColormap.from_list(
    "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
)

# norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one = Normalize(vmin=0, vmax=1)

cmap_weighting = get_cmap("Reds")
cmap_edge = get_cmap("Greys")
cmap_multi = get_cmap("plasma")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

alpha_quick, alpha_normal, alpha_lagard = 0.9, 0.7, 0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"


min_val = 1e-3
dpi_save = 600  # 1200

alpha_val = 0.25
size_points = 5

RUN = 0
SINGLE_COMPARE_IDENTITY = 1
ONE_PARAM = 0
TWO_PARAM_ONE_SEED = 0
bifurcation_plot_data_analysis = 0

fileName = "results/" + "bifurcation_SINGLE_COMPARE_IDENTITY_16_52_02__03_01_2023"#one_param_sweep_multi_18_06_16__16_12_2022"
"""
{'save_timeseries_data': 0, 'degroot_aggregation': 1, 'network_structure': 'small_world', 'alpha_change': 2.0, 'guilty_individuals': 0, 'moral_licensing': 0, 'immutable_green_fountains': 1, 'polarisation_test': 0, 'total_time': 3000, 'delta_t': 1.0, 'phi_lower': 0.01, 'phi_upper': 0.05, 'compression_factor': 10, 'seed_list': [1, 2, 3, 4, 5], 'N': 200, 'M': 3, 'K': 20, 'prob_rewire': 0.1, 'set_seed': 1, 'culture_momentum_real': 1000, 'learning_error_scale': 0.02, 'discount_factor': 0.95, 'homophily': 0.95, 'homophilly_rate': 1, 'confirmation_bias': 100.0, 'a_attitude': 1, 'b_attitude': 1, 'a_threshold': 1, 'b_threshold': 1, 'action_observation_I': 0.0, 'action_observation_S': 0.0, 'green_N': 
0, 'guilty_individual_power': 0, 'time_steps_max': 3000}
"""


if __name__ == "__main__":
    if SINGLE_COMPARE_IDENTITY:
        ############################
        if RUN:
            #############################################################
            ###FIRST RUN WITH IDENTITY (BEHAVIORAL INTERDEPENDANCE), ALPHA  = 1.0
            property_varied_identity = "confirmation_bias"
            property_varied_title_identity = "Confirmation bias $\theta$"
            param_min_identity = 0.0
            param_max_identity = 100.0  # 50.0
            reps_identity = 500
            title_list_identity = ["Bifurcation"]
            #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
            property_values_list_identity = np.linspace(param_min_identity,param_max_identity, reps_identity)
            print("property_values_list_identity ", property_values_list_identity )

            f = open("constants/base_params.json")
            base_params = json.load(f)
            base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
            base_params["alpha_change"] = 1.0

            params_list_identity = produce_param_list(base_params, property_values_list_identity, property_varied_identity)
            results_culture_lists_identity = one_seed_culture_data_run(params_list_identity)#list of lists lists [param set up, stochastic, cluster]

            #####################################################################
            ####NO IDENTITY, ALPHA  = 2.0
            property_varied_no_identity = "confirmation_bias"
            property_varied_title_no_identity = "Confirmation bias $\theta$"
            param_min_no_identity = 0.0
            param_max_no_identity = 100.0  # 50.0
            reps_no_identity = 500
            title_list_no_identity = ["Bifurcation"]
            #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
            property_values_list_no_identity = np.linspace(param_min_no_identity,param_max_no_identity, reps_no_identity)
            print("property_values_list_no_identity ", property_values_list_no_identity )

            base_params["alpha_change"] = 2.0

            params_list_no_identity = produce_param_list(base_params, property_values_list_no_identity, property_varied_no_identity)
            results_culture_lists_no_identity = one_seed_culture_data_run(params_list_no_identity)#list of lists lists [param set up, stochastic, cluster]

            ############################################################################

            root = "bifurcation_SINGLE_COMPARE_IDENTITY"
            fileName = produce_name_datetime(root)
            print("fileName: ", fileName)

            createFolder(fileName)

            save_object(base_params, fileName + "/Data", "base_params")

            save_object(results_culture_lists_identity, fileName + "/Data", "results_culture_lists_identity")
            save_object(property_varied_identity, fileName + "/Data", "property_varied_identity")
            save_object(property_varied_title_identity, fileName + "/Data", "property_varied_title_identity")
            save_object(param_min_identity, fileName + "/Data", "param_min_identity")
            save_object(param_max_identity, fileName + "/Data", "param_max_identity")
            save_object(title_list_identity, fileName + "/Data", "title_list_identity")
            save_object(property_values_list_identity, fileName + "/Data", "property_values_list_identity")

            save_object(results_culture_lists_no_identity, fileName + "/Data", "results_culture_lists_no_identity")
            save_object(property_varied_no_identity, fileName + "/Data", "property_varied_no_identity")
            save_object(property_varied_title_no_identity, fileName + "/Data", "property_varied_title_no_identity")
            save_object(param_min_no_identity, fileName + "/Data", "param_min_no_identity")
            save_object(param_max_no_identity, fileName + "/Data", "param_max_no_identity")
            save_object(title_list_no_identity, fileName + "/Data", "title_list_no_identity")
            save_object(property_values_list_no_identity, fileName + "/Data", "property_values_list_no_identity")
        else:
            base_params = load_object(fileName + "/Data", "base_params")
            #print(base_params)
            #quit()
            results_culture_lists_identity = load_object(fileName + "/Data", "results_culture_lists_identity")
            property_varied_identity = load_object(fileName + "/Data", "property_varied_identity")
            property_varied_title_identity = load_object(fileName + "/Data", "property_varied_title_identity")
            param_min_identity = load_object(fileName + "/Data", "param_min_identity")
            param_max_identity = load_object(fileName + "/Data", "param_max_identity")
            title_list_identity = load_object(fileName + "/Data", "title_list_identity")
            property_values_list_identity = load_object(fileName + "/Data", "property_values_list_identity")
            
            results_culture_lists_no_identity = load_object(fileName + "/Data", "results_culture_lists_no_identity")
            property_varied_no_identity = load_object(fileName + "/Data", "property_varied_no_identity")
            property_varied_title_no_identity = load_object(fileName + "/Data", "property_varied_title_no_identity")
            param_min_no_identity = load_object(fileName + "/Data", "param_min_no_identity")
            param_max_no_identity = load_object(fileName + "/Data", "param_max_no_identity")
            title_list_no_identity = load_object(fileName + "/Data", "title_list_no_identity")
            property_values_list_no_identity = load_object(fileName + "/Data", "property_values_list_no_identity")

        ###WORKING
        N = base_params["N"]

        if bifurcation_plot_data_analysis:
            no_samples = 10000
            identity_space = np.linspace(0, 1,no_samples)
            bandwidth = 0.01

            cluster_pos_matrix_list_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_identity[i]),identity_space,bandwidth) for i in range(len(property_values_list_identity))] 
            cluster_pos_matrix_list_no_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_no_identity[i]),identity_space,bandwidth) for i in range(len(property_values_list_no_identity))] 
            
            save_object(cluster_pos_matrix_list_no_identity, fileName + "/Data", "cluster_pos_matrix_list_no_identity")
            save_object(cluster_pos_matrix_list_identity, fileName + "/Data", "cluster_pos_matrix_list_identity")

            save_object(identity_space, fileName + "/Data", "identity_space")
            save_object(bandwidth, fileName + "/Data", "bandwidth")
        else:
            cluster_pos_matrix_list_no_identity = load_object(fileName + "/Data", "cluster_pos_matrix_list_no_identity")
            cluster_pos_matrix_list_identity = load_object(fileName + "/Data", "cluster_pos_matrix_list_identity")
            #print("cluster_pos_matrix",cluster_pos_matrix_list)

        #bifurcation_plot(fileName,cluster_pos_matrix,property_values_list, dpi_save)
        #bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        #bifurcation_heat_map_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        bifurcation_plot_culture_or_not(fileName,cluster_pos_matrix_list_identity,cluster_pos_matrix_list_no_identity,property_values_list_identity,property_values_list_no_identity, dpi_save)

    elif ONE_PARAM:
        ############################
        if RUN:
            seed_list_set = list(range(20))
            print(seed_list_set)
            property_varied = "confirmation_bias"
            property_varied_title = "Confirmation bias $\theta$"
            param_min = -10.0
            param_max = 100.0  # 50.0
            reps = 220
            title_list = ["Bifurcation"]
            #title_list = [r"Confirmation bias $\theta$ = 1.0", r"Confirmation bias $\theta$ = 10.0", r"Confirmation bias $\theta$ = 25.0", r"Confirmation bias $\theta$ = 50.0"]
            property_values_list = np.linspace(param_min,param_max, reps)
            print("property_values_list ", property_values_list )

            f = open("constants/base_params.json")
            base_params = json.load(f)
            base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
            base_params["seed_list"] = seed_list_set

            root = "bifurcation_one_param"
            fileName = produce_name_datetime(root)
            print("fileName: ", fileName)

            params_list = produce_param_list(base_params, property_values_list, property_varied)

            results_culture_lists = culture_data_run(params_list)#list of lists lists [param set up, stochastic, cluster]

            createFolder(fileName)

            save_object(results_culture_lists, fileName + "/Data", "results_culture_lists")
            save_object(base_params, fileName + "/Data", "base_params")
            save_object(property_varied, fileName + "/Data", "property_varied")
            save_object(property_varied_title, fileName + "/Data", "property_varied_title")
            save_object(param_min, fileName + "/Data", "param_min")
            save_object(param_max, fileName + "/Data", "param_max")
            save_object(title_list, fileName + "/Data", "title_list")
            save_object(property_values_list, fileName + "/Data", "property_values_list")
        else:
            results_culture_lists = load_object(fileName + "/Data", "results_culture_lists")
            base_params = load_object(fileName + "/Data", "base_params")
            property_varied = load_object(fileName + "/Data", "property_varied")
            property_varied_title = load_object(fileName + "/Data", "property_varied_title")
            param_min = load_object(fileName + "/Data", "param_min")
            param_max = load_object(fileName + "/Data", "param_max")
            title_list = load_object(fileName + "/Data", "title_list")
            property_values_list = load_object(fileName + "/Data", "property_values_list")

        ###WORKING
        N = base_params["N"]
        seed_list = base_params["seed_list"]

        if bifurcation_plot_data_analysis:
            no_samples = 10000
            identity_space = np.linspace(0, 1,no_samples)
            bandwidth = 0.01

            cluster_pos_matrix_list = [[calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists[i][j]),identity_space,bandwidth) for i in range(len(property_values_list))] for j in range(len(seed_list))]
            #print("cluster_pos_matrix_list",cluster_pos_matrix_list,len(cluster_pos_matrix_list))
            # each entry is a matrix of position of identity clusters, corresponding to one stochastic run
            
            save_object(cluster_pos_matrix_list, fileName + "/Data", "cluster_pos_matrix_list")
            save_object(identity_space, fileName + "/Data", "identity_space")
            save_object(bandwidth, fileName + "/Data", "bandwidth")
        else:
            cluster_pos_matrix_list = load_object(fileName + "/Data", "cluster_pos_matrix_list")
            #print("cluster_pos_matrix",cluster_pos_matrix_list)

        #bifurcation_plot(fileName,cluster_pos_matrix,property_values_list, dpi_save)
        #bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        bifurcation_heat_map_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
    elif TWO_PARAM_ONE_SEED:
        if RUN:
            f = open("constants/base_params.json")
            base_params = json.load(f)
            base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])
            #base_params["seed_list"] = seed_list_set

            root = "bifurcation_one_param"
            fileName = produce_name_datetime(root)
            print("fileName: ", fileName)

            variable_parameters_dict = {
                "row":{"property":"confirmation_bias","min":-10, "max":100 , "title": "Confirmation bias, $\\theta$","divisions": "linear", "reps": 10},  
                "col":{"property":"a_attitude","min":0.05, "max": 3, "title": "Attitude Beta $a$","divisions": "linear", "reps": 10}, 
            }

            variable_parameters_dict = generate_vals_variable_parameters_and_norms(
                variable_parameters_dict
            )
            params_list = produce_param_list_n_double(base_params, variable_parameters_dict)

            results_culture_lists = one_seed_culture_data_run(params_list)#list of lists lists [param set up, stochastic, cluster]

            createFolder(fileName)

            # save the data and params_list
            save_object(variable_parameters_dict, fileName + "/Data", "variable_parameters_dict")
            save_object(base_params, fileName + "/Data", "base_params")
            save_object(results_culture_lists,fileName + "/Data","results_culture_lists") 
        else:
            variable_parameters_dict = load_object(fileName + "/Data", "variable_parameters_dict")
            base_params = load_object(fileName + "/Data", "base_params")
            results_culture_lists = load_object(fileName + "/Data","results_culture_lists")

        ###WORKING
        N = base_params["N"]
        seed_list = base_params["seed_list"]

        if bifurcation_plot_data_analysis:
            no_samples = 10000
            identity_space = np.linspace(0, 1,no_samples)
            bandwidth = 0.01

            cluster_pos_matrix_list = [[calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists[i][j]),identity_space,bandwidth) for j in range(len(variable_parameters_dict["col"]["vals"]))] for i in range(len(variable_parameters_dict["row"]["vals"]))]
            #print("cluster_pos_matrix_list",cluster_pos_matrix_list,len(cluster_pos_matrix_list))
            # each entry is a matrix of position of identity clusters, corresponding to one stochastic run
            
            save_object(cluster_pos_matrix_list, fileName + "/Data", "cluster_pos_matrix_list")
            save_object(identity_space, fileName + "/Data", "identity_space")
            save_object(bandwidth, fileName + "/Data", "bandwidth")
        else:
            cluster_pos_matrix_list = load_object(fileName + "/Data", "cluster_pos_matrix_list")
            #print("cluster_pos_matrix",cluster_pos_matrix_list)

        #bifurcation_plot(fileName,cluster_pos_matrix,property_values_list, dpi_save)
        #bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        bifurcation_plot_one_seed_two_params(fileName,cluster_pos_matrix_list,variable_parameters_dict,cmap_multi,dpi_save)
    plt.show()
