"""Runs a single simulation to produce data which is saved and plotted 
A module that use dictionary of data for the simulation run. The single shot simualtion is run
for a given intial set seed. The desired plots are then produced and saved.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
import numpy as np
from resources.utility import ( 
    load_object,
    save_object,
)
from resources.plot import (
    plot_culture_timeseries,
    plot_cum_link_change_per_agent,
    plot_value_timeseries,
    plot_attitude_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_culture_timeseries,
    plot_green_adoption_timeseries,
    cluster_estimation_plot,
    plot_alpha_group,
    plot_cluster_culture_time_series,
    print_live_initial_culture_network,
    live_animate_culture_network_and_weighting,
)


def main(
    fileName = "results/single_shot_11_52_34__05_01_2023",
    PLOT_NAME = "INDIVIDUAL",
    node_size = 50,
    fps = 5,
    interval = 50,
    layout = "circular",
    round_dec = 3,
    dpi_save = 2000,
    no_samples = 10000,
    bandwidth = 0.01,
    shuffle_colours = True,
    animation_save_bool = 0
    ) -> None: 

    cmap_multi = get_cmap("plasma")
    cmap_edge = get_cmap("Greys")

    norm_zero_one = Normalize(vmin=0, vmax=1),
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    ),

    Data = load_object(fileName + "/Data", "social_network")

    ###PLOTS
    if PLOT_NAME == "INDIVIDUAL":
        plot_culture_timeseries(fileName, Data, dpi_save)
        plot_value_timeseries(fileName,Data,dpi_save)
        plot_attitude_timeseries(fileName,Data,dpi_save)
        plot_cum_link_change_per_agent(fileName,Data,dpi_save)
        print_live_initial_culture_network(fileName,Data,dpi_save,layout,norm_zero_one,cmap,node_size,round_dec)
    elif PLOT_NAME == "NETWORK":
        plot_green_adoption_timeseries(fileName, Data, dpi_save)
        plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save)
        plot_weighting_matrix_convergence_timeseries(fileName, Data, dpi_save)
        plot_cultural_range_timeseries(fileName, Data, dpi_save)
        plot_average_culture_timeseries(fileName,Data,dpi_save)
    elif PLOT_NAME == "ANIMATION":
        live_animate_culture_network_and_weighting(    
            fileName,
            Data,
            layout,
            cmap,
            node_size,
            interval,
            fps,
            norm_zero_one,
            round_dec,
            cmap_edge,
            animation_save_bool
        )
    elif PLOT_NAME == "CLUSTERING":
        cluster_estimation_plot(Data,s,bandwidth)

        s = np.linspace(0, 1,no_samples)
        clusters_index_lists, cluster_example_identity_list = plot_alpha_group(fileName, Data, dpi_save,s, bandwidth,cmap_multi, round_dec,norm_zero_one )#set it using input value
        save_object(clusters_index_lists, fileName + "/Data", "clusters_index_lists")
        save_object(cluster_example_identity_list, fileName + "/Data", "cluster_example_identity_list")

        plot_cluster_culture_time_series(fileName, Data, dpi_save,clusters_index_lists, cluster_example_identity_list, cmap_multi,norm_zero_one, shuffle_colours)#haS TO BE RUN TOGETHER

    plt.show()
