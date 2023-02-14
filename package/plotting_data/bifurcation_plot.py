"""Produce bifurcation plots

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from package.resources.utility import save_object,load_object,calc_pos_clusters_set_bandwidth
from package.resources.plot import (
    bifurcation_plot_culture_or_not,
)

def main(
    fileName = "results/" + "bifurcation_SINGLE_COMPARE_IDENTITY_16_52_02__03_01_2023",
    bifurcation_plot_data_analysis = 0,
    dpi_save = 1200,
    no_samples = 10000,
    bandwidth = 0.01
    ) -> None: 

    results_culture_lists_identity = load_object(fileName + "/Data", "results_culture_lists_identity")
    results_culture_lists_no_identity = load_object(fileName + "/Data", "results_culture_lists_no_identity")
    property_values_list = load_object(fileName + "/Data", "property_values_list")

    if bifurcation_plot_data_analysis:#If you want to perform the cluster analysis for different bandwidths
        identity_space = np.linspace(0, 1,no_samples)

        cluster_pos_matrix_list_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_identity[i]),identity_space,bandwidth) for i in range(len(property_values_list))] 
        cluster_pos_matrix_list_no_identity = [calc_pos_clusters_set_bandwidth(np.asarray(results_culture_lists_no_identity[i]),identity_space,bandwidth) for i in range(len(property_values_list))] 
        
        save_object(cluster_pos_matrix_list_no_identity, fileName + "/Data", "cluster_pos_matrix_list_no_identity")
        save_object(cluster_pos_matrix_list_identity, fileName + "/Data", "cluster_pos_matrix_list_identity")

        save_object(identity_space, fileName + "/Data", "identity_space")
        save_object(bandwidth, fileName + "/Data", "bandwidth")
    else:
        cluster_pos_matrix_list_no_identity = load_object(fileName + "/Data", "cluster_pos_matrix_list_no_identity")
        cluster_pos_matrix_list_identity = load_object(fileName + "/Data", "cluster_pos_matrix_list_identity")

    bifurcation_plot_culture_or_not(fileName,cluster_pos_matrix_list_identity,cluster_pos_matrix_list_no_identity,property_values_list, dpi_save)

    plt.show()
