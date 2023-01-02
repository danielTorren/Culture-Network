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
from resources.utility import createFolder,produce_name_datetime,save_object,load_object,calc_pos_clusters_set_bandwidth, get_cluster_list
from resources.run import parallel_run, parallel_run_sa,culture_data_run
from resources.plot import (
    bifurcation_plot,
    bifurcation_plot_stochastic,
)
from resources.multi_run_single_param import (
    produce_param_list,
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

RUN = 1
bifurcation_plot_data_analysis = 1

fileName = "results/" + "one_param_sweep_single_23_02_15__01_01_2023"#one_param_sweep_multi_18_06_16__16_12_2022"

if __name__ == "__main__":

    ############################
    if RUN:
        seed_list_set = [1,2,3,4,5,6,7,8,9,10]
        property_varied = "confirmation_bias"
        property_varied_title = "Confirmation bias $\theta$"
        param_min = 0.0
        param_max = 100.0  # 50.0
        reps = 50
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
        print("cluster_pos_matrix_list",cluster_pos_matrix_list,len(cluster_pos_matrix_list))
        # each entry is a matrix of position of identity clusters, corresponding to one stochastic run
        
        save_object(cluster_pos_matrix_list, fileName + "/Data", "cluster_pos_matrix_list")
        save_object(identity_space, fileName + "/Data", "identity_space")
        save_object(bandwidth, fileName + "/Data", "bandwidth")
    else:
        cluster_pos_matrix_list = load_object(fileName + "/Data", "cluster_pos_matrix_list")
        print("cluster_pos_matrix",cluster_pos_matrix_list)

    #bifurcation_plot(fileName,cluster_pos_matrix,property_values_list, dpi_save)
    bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
    
    plt.show()
