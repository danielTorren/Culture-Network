"""Plots a single simulation to produce data which is saved and plotted 

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
)
from plotting_data.identity_frequency_plot import calc_groups
from resources.plot import (
    plot_culture_timeseries,
    plot_value_timeseries,
    plot_attitude_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_culture_timeseries,
    plot_joint_cluster_micro,
    print_live_initial_culture_network,
    live_animate_culture_network_weighting_matrix,
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
    cmap_weighting = get_cmap("Reds")

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    )

    Data = load_object(fileName + "/Data", "social_network")

    ###PLOTS
    if PLOT_NAME == "INDIVIDUAL":
        plot_culture_timeseries(fileName, Data, dpi_save)
        plot_value_timeseries(fileName,Data,dpi_save)
        plot_attitude_timeseries(fileName,Data,dpi_save)
        print_live_initial_culture_network(
            fileName,
            Data,
            dpi_save,
            layout,
            norm_zero_one,
            cmap,
            node_size
        )
    elif PLOT_NAME == "NETWORK":
        plot_total_carbon_emissions_timeseries(fileName, Data, dpi_save)
        plot_weighting_matrix_convergence_timeseries(fileName, Data, dpi_save)
        plot_cultural_range_timeseries(fileName, Data, dpi_save)
        plot_average_culture_timeseries(fileName,Data,dpi_save)
    elif PLOT_NAME == "ANIMATION":
        anim = live_animate_culture_network_weighting_matrix(
            fileName,
            Data,
            cmap_weighting,
            interval,
            fps,
            round_dec,
            layout,
            cmap,
            node_size,
            norm_zero_one,
            animation_save_bool
        )

    elif PLOT_NAME == "CLUSTERING":
        s = np.linspace(0, 1,no_samples)
        clusters_index_lists, cluster_example_identity_list, vals_time_data = calc_groups(Data,s, bandwidth)#set it using input value
        plot_joint_cluster_micro(fileName, Data, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, 0, bandwidth,cmap_multi, norm_zero_one,shuffle_colours)


    plt.show()
