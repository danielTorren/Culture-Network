"""Run multiple single simulations varying a single parameter
A module that use input data to generate data from multiple social networks varying a single property
between simulations so that the differences may be compared. These multiple runs can either be single 
shot runs or taking the average over multiple runs.

TWO MODES 
    Single parameters can be varied to cover a list of points. This can either be done in SINGLE = True where individual 
    runs are used as the output and gives much greater variety of possible plots but all these plots use the same initial
    seed. Alternatively can be run such that multiple averages of the simulation are produced and then the data accessible 
    is the emissions, mean identity, variance of identity and the coefficient of variance of identity.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
from resources.utility import save_object,load_object,calc_pos_clusters_set_bandwidth
from resources.plot import (
    live_multirun_diagram_mean_coefficient_variance,
    live_print_culture_timeseries,
    plot_average_culture_comparison,
    plot_carbon_emissions_total_comparison,
    plot_weighting_matrix_convergence_comparison,
    plot_average_culture_no_range_comparison,
    plot_live_link_change_comparison,
    plot_live_cum_link_change_comparison,
    plot_live_link_change_per_agent_comparison,
    plot_live_cum_link_change_per_agent_comparison,
    live_multirun_diagram_mean_coefficient_variance,
    print_live_intial_culture_networks,
    prints_init_weighting_matrix,
    prints_final_weighting_matrix,
    live_print_culture_timeseries_with_weighting,
    print_live_intial_culture_networks_and_culture_timeseries,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    live_compare_plot_animate_behaviour_scatter,
    live_varaince_timeseries,
    plot_alpha_group_multi,
    plot_cluster_culture_time_series_multi,
    plot_last_culture_vector_matrix,
    plot_last_culture_vector_joy,
    plot_last_culture_vector_joy_hist,
    bifurcation_plot,
)

def main(
    fileName = "results/" + "one_param_sweep_single_17_43_28__31_01_2023",
    PLOT_NAME = "SINGLE",
    GRAPH_TYPE = 0,
    ANIMATION = 0,
    bifurcation_plot_data_analysis = 0,
    Data_list_bool = 1,
    node_size = 100,
    round_dec = 2,
    dpi_save = 600,
    no_samples = 10000,
    bandwidth = 0.01,
    fps = 5,
    interval = 50,
    nrows_plot = 2, #leave as 1 for alpha and homophily plots, but change for network!
    ncols_plot = 3,  # due to screen ratio want more cols than rows usually
    ) -> None: 

    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    ),
    norm_zero_one = Normalize(vmin=0, vmax=1),
    cmap_weighting = get_cmap("Reds"),
    cmap_edge = get_cmap("Greys"),
    cmap_multi = get_cmap("plasma"),

    ############################
    if PLOT_NAME == "SINGLE":

        data_list = load_object(fileName + "/Data", "data_list")
        params = load_object(fileName + "/Data", "base_params")
        property_varied = load_object(fileName + "/Data", "property_varied")
        property_varied_title = load_object(fileName + "/Data", "property_varied_title")
        param_min = load_object(fileName + "/Data", "param_min")
        param_max = load_object(fileName + "/Data", "param_max")
        title_list = load_object(fileName + "/Data", "title_list")
        property_values_list = load_object(fileName + "/Data", "property_values_list")

        dataName = fileName + "/Data"
        ###WORKING

        if GRAPH_TYPE == 0:
            #FOR POLARISATION A,B PLOT - NEED TO SET self.b_attitude = parameters["a_attitude"] in NETWORK
            live_print_culture_timeseries(fileName, data_list, property_varied, title_list, nrows_plot, ncols_plot, dpi_save)
        elif GRAPH_TYPE == 1:
            ################
            #FOR ALPHA CHANGE PLOT
            live_print_culture_timeseries_with_weighting(fileName, data_list, property_varied, title_list, nrows_plot, ncols_plot, dpi_save, cmap_weighting)
        elif GRAPH_TYPE == 2:
            ###############################
            #FOR HOMOPHILY PLOT
            layout = ["circular","circular", "circular"]
            print_live_intial_culture_networks_and_culture_timeseries(fileName, data_list, dpi_save, property_values_list, property_varied_title, ncols_plot, layout, norm_zero_one, cmap, node_size,round_dec)
        elif GRAPH_TYPE == 3:
            layout = ["circular","circular", "circular", "spring", "spring", "spring"]
            print_live_intial_culture_networks_and_culture_timeseries(fileName, data_list, dpi_save, property_values_list, property_varied_title, ncols_plot, layout, norm_zero_one, cmap, node_size,round_dec)
        elif GRAPH_TYPE == 4:
            live_print_culture_timeseries(fileName, data_list, property_varied, title_list, nrows_plot, ncols_plot, dpi_save)
            plot_carbon_emissions_total_comparison(fileName, data_list, dpi_save, property_values_list, property_varied_title, round_dec)
        elif GRAPH_TYPE == 5:
            s = np.linspace(0, 1,no_samples)
            auto_bandwidth = False
            clusters_index_lists_list, cluster_example_identity_list_list = plot_alpha_group_multi(fileName, data_list, dpi_save,s, auto_bandwidth, bandwidth,cmap_multi, norm_zero_one ,nrows_plot, ncols_plot,title_list)#set it using input value
            save_object(clusters_index_lists_list, fileName + "/Data", "clusters_index_lists_list")
            save_object(cluster_example_identity_list_list, fileName + "/Data", "cluster_example_identity_list_list")

            clusters_index_lists_list = load_object(dataName,"clusters_index_lists_list")
            cluster_example_identity_list_list = load_object(dataName,"cluster_example_identity_list_list")
            shuffle_colours = True
            plot_cluster_culture_time_series_multi(fileName, data_list, dpi_save,clusters_index_lists_list, cluster_example_identity_list_list, cmap_multi,norm_zero_one, shuffle_colours ,nrows_plot, ncols_plot,title_list)#haS TO BE RUN TOGETHER
        elif GRAPH_TYPE == (6 or 7):
            plot_last_culture_vector_matrix(fileName, data_list, dpi_save, property_varied, property_varied_title, property_values_list)
            plot_last_culture_vector_joy(fileName, data_list, dpi_save, property_varied, property_varied_title, property_values_list,cmap_multi,Data_list_bool)
            plot_last_culture_vector_joy_hist(fileName, data_list, dpi_save, property_varied, property_varied_title, property_values_list,cmap_multi,Data_list_bool)
            plot_carbon_emissions_total_comparison(fileName, data_list, dpi_save, property_values_list, property_varied_title, round_dec)
        elif GRAPH_TYPE == 8:
            if bifurcation_plot_data_analysis:
                identity_space = np.linspace(0, 1,no_samples)
                N = data_list[0].N
                cluster_pos_matrix = [calc_pos_clusters_set_bandwidth(np.asarray(i.culture_list),identity_space,bandwidth) for i in data_list]
                # this is a matrix of position of identity clusters within in 
                save_object(cluster_pos_matrix, fileName + "/Data", "cluster_pos_matrix")
            else:
                cluster_pos_matrix = load_object(fileName + "/Data", "cluster_pos_matrix")
            
            bifurcation_plot(fileName,cluster_pos_matrix,property_values_list, dpi_save)
        else:
            live_print_culture_timeseries(fileName, data_list, property_varied, title_list, nrows_plot, ncols_plot, dpi_save)
            plot_average_culture_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            plot_carbon_emissions_total_comparison(fileName, data_list, dpi_save, property_values_list, property_varied, round_dec)
            plot_weighting_matrix_convergence_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            plot_average_culture_no_range_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            plot_live_link_change_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            plot_live_cum_link_change_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            plot_live_link_change_per_agent_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            plot_live_cum_link_change_per_agent_comparison(fileName, data_list, dpi_save,property_values_list, property_varied,round_dec)
            live_varaince_timeseries( fileName,data_list,property_varied,property_varied_title,property_values_list,dpi_save,)
            live_multirun_diagram_mean_coefficient_variance(fileName,data_list,property_varied,property_values_list,property_varied_title,cmap,dpi_save,norm_zero_one,)
            print_live_intial_culture_networks(fileName, data_list, dpi_save, property_values_list, property_varied, nrows_plot, ncols_plot , layout, norm_zero_one, cmap, node_size,round_dec)
            prints_init_weighting_matrix(fileName, data_list, dpi_save,nrows_plot, ncols_plot, cmap_weighting,property_values_list, property_varied,round_dec)
            prints_final_weighting_matrix(fileName, data_list, dpi_save,nrows_plot, ncols_plot, cmap_weighting,property_values_list, property_varied,round_dec)
            live_print_culture_timeseries_with_weighting(fileName, data_list, property_varied, title_list, nrows_plot, ncols_plot, dpi_save, cmap_weighting)
            print_live_intial_culture_networks_and_culture_timeseries(fileName, data_list, dpi_save, property_values_list, property_varied_title, ncols_plot, layout, norm_zero_one, cmap, node_size,round_dec)
            if ANIMATION:
                ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols_plot, nrows_plot,property_varied_title,property_values_list)
                ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows_plot, ncols_plot,property_varied_title,property_values_list)
                ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, nrows_plot, ncols_plot,property_varied_title,property_values_list)
                ani_e = live_compare_plot_animate_behaviour_scatter(fileName,data_list,norm_zero_one, cmap, nrows_plot, ncols_plot,property_varied, property_values_list,interval, fps,round_dec)
    elif PLOT_NAME == "MULTI":
        property_varied = load_object(fileName + "/Data", "property_varied")
        property_varied_title = load_object(fileName + "/Data", "property_varied_title")
        title_list = load_object(fileName + "/Data", "title_list")
        property_values_list = load_object(fileName + "/Data", "property_values_list")

        if GRAPH_TYPE == 6 or 7:
            results_culture_lists = load_object( fileName + "/Data", "results_culture_lists")
        else:
            results_emissions = load_object(fileName + "/Data", "results_emissions")
            results_mu = load_object(fileName + "/Data", "results_mu")
            results_var = load_object(fileName + "/Data", "results_var")
            results_coefficient_of_variance = load_object(fileName + "/Data", "results_coefficient_of_variance")
            results_emissions_change = load_object( fileName + "/Data", "results_emissions_change")
            

        if GRAPH_TYPE == 6 or 7:
            Data_list_bool = 0
            plot_last_culture_vector_matrix(fileName, data_list, dpi_save, property_varied, property_varied_title, property_values_list)
            plot_last_culture_vector_joy(fileName, results_culture_lists, dpi_save, property_varied, property_varied_title, property_values_list,cmap_multi,Data_list_bool)
            plot_last_culture_vector_joy_hist(fileName, results_culture_lists, dpi_save, property_varied, property_varied_title, property_values_list,cmap_multi,Data_list_bool)
            plot_carbon_emissions_total_comparison(fileName, data_list, dpi_save, property_values_list, property_varied_title, round_dec)

    plt.show()
