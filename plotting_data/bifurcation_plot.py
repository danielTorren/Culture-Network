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
from resources.plot import (
    bifurcation_plot,
    bifurcation_plot_stochastic,
    bifurcation_plot_one_seed_two_params,
    bifurcation_heat_map_stochastic,
    bifurcation_plot_culture_or_not,
)

def main(
    fileName = "results/" + "bifurcation_SINGLE_COMPARE_IDENTITY_16_52_02__03_01_2023",
    PLOT_NAME = "SINGLE_COMPARE_IDENTITY",
    bifurcation_plot_data_analysis = 0,
    dpi_save = 1200,
    ) -> None: 
    cmap_multi = get_cmap("plasma"),
    if PLOT_NAME == "SINGLE":

        base_params = load_object(fileName + "/Data", "base_params")
        results_culture_lists_identity = load_object(fileName + "/Data", "results_culture_lists_identity")
        property_values_list_identity = load_object(fileName + "/Data", "property_values_list_identity")
        
        results_culture_lists_no_identity = load_object(fileName + "/Data", "results_culture_lists_no_identity")
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

        bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        bifurcation_heat_map_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        bifurcation_plot_culture_or_not(fileName,cluster_pos_matrix_list_identity,cluster_pos_matrix_list_no_identity,property_values_list_identity,property_values_list_no_identity, dpi_save)
    elif PLOT_NAME == "MULTI":

        results_culture_lists = load_object(fileName + "/Data", "results_culture_lists")
        base_params = load_object(fileName + "/Data", "base_params")
        property_values_list = load_object(fileName + "/Data", "property_values_list")

        ###WORKING
        N = base_params["N"]
        seed_list = base_params["seed_list"]

        if bifurcation_plot_data_analysis:
            no_samples = 10000
            identity_space = np.linspace(0, 1,no_samples)
            bandwidth = 0.01

            cluster_pos_matrix_list = [[calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists[i][j]),identity_space,bandwidth) for i in range(len(property_values_list))] for j in range(len(seed_list))]

            # each entry is a matrix of position of identity clusters, corresponding to one stochastic run
            
            save_object(cluster_pos_matrix_list, fileName + "/Data", "cluster_pos_matrix_list")
            save_object(identity_space, fileName + "/Data", "identity_space")
            save_object(bandwidth, fileName + "/Data", "bandwidth")
        else:
            cluster_pos_matrix_list = load_object(fileName + "/Data", "cluster_pos_matrix_list")

        bifurcation_plot_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
        bifurcation_heat_map_stochastic(fileName,cluster_pos_matrix_list,property_values_list, seed_list,cmap_multi,dpi_save)
    plt.show()
