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
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.cm import get_cmap
from resources.utility import createFolder
from resources.run import parallel_run, parallel_run_sa
from resources.plot import (
    live_multirun_diagram_mean_coefficient_variance,
    live_average_multirun_diagram_mean_coefficient_variance,
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
    draw_networks,
    live_print_culture_timeseries_varynetwork_structure,
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

SINGLE = 1
GRAPH_TYPE = 4


if __name__ == "__main__":
    """The number of rows and cols set the number of experiments ie 4 rows and 3 cols gives 12 experiments"""
    nrows = 2 #leave as 1 for alpha and homophily plots, but change for network!
    ncols = 2  # due to screen ratio want more cols than rows usually
    reps = nrows * ncols  # make multiples of the number of cores for efficieny

    ############################
    if GRAPH_TYPE == 0:
        #FOR POLARISATION A,B PLOT - NEED TO SET self.b_attitude = parameters["a_attitude"] in NETWORK
        property_varied = "a_attitude"
        property_varied_title = "Attitude Beta parameters, (a,b)"
        param_min = 0.05
        param_max = 2.0  # 50.0
        property_values_list = np.asarray([0.05, 0.5, 2.0])# FOR ALPHA
        title_list = [r"Attitude Beta parameters, (a,b) = 0.05, Confirmation bias, $\theta$ = 40", r"Attitude Beta parameters, (a,b) = 0.5, Confirmation bias, $\theta$ = 20", r"Attitude Beta parameters, (a,b) = 2, Confirmation bias, $\theta$ = 10"]
    elif GRAPH_TYPE == 1:
        ################
        #FOR ALPHA CHANGE PLOT
        property_varied = "alpha_change"
        property_varied_title = "alpha_change"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Static uniform $\alpha_{n,k}$", r"Static culturally determined $\alpha_{n,k}$", r"Dynamic culturally determined $\alpha_{n,k}$"]
        property_values_list = np.asarray([0.0, 0.5, 1.0])# FOR ALPHA
    elif GRAPH_TYPE == 2:
        ###############################
        #FOR HOMOPHILY PLOT
        property_varied = "homophily"
        property_varied_title = "Attribute homophily, $h$"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Homophily, h = 0.0", r"Homophily, h = 0.5", r"Homophily, h = 1.0"]
        property_values_list = np.asarray([0.2, 0.6, 1.0])
    elif GRAPH_TYPE == 3:
        ###############################
        #FOR NETOWRK STRUCTURE HOMOPHILY PLOT
        property_varied = "homophily"
        property_varied_title = "Attribute homophily, $h$"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Small world, Homophily, h = 0.0", r"Small world, Homophily, h = 0.5", r"Small world, Homophily, h = 1.0",r"Scale free, Homophily, h = 0.0", r"Scale free, Homophily, h = 0.5", r"Scale free, Homophily, h = 1.0"]
        property_values_list = np.asarray([0.2, 0.6, 1.0, 0.2, 0.6, 1.0])
    elif GRAPH_TYPE == 4:
        property_varied = "guilty_individual_power"
        property_varied_title = "Identity power"
        param_min = 1.0
        param_max = 20.0  # 50.0
        title_list = [r"Identity power = 1.0", r"Identity power = 5.0", r"Identity power = 10.0", r"Identity power = 20.0"]
        property_values_list = np.asarray([1.0, 5.0, 10.0, 20.0])
        
    f = open("constants/base_params.json")
    params = json.load(f)
    params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

    if SINGLE:
        # SINGLE SHOT RUNS NO AVERAGING OVER STOCHASTIC EFFECTS
        fileName = "results/network_struct_%s_variation_%s_%s_%s_%s_%s_%s" % (#CHANGE THIS BACK!!!!
            property_varied,
            str(params["N"]),
            str(params["time_steps_max"]),
            str(params["K"]),
            str(param_min),
            str(param_max),
            str(reps),
        )
        print("fileName: ", fileName)
        createFolder(fileName)

        if GRAPH_TYPE == 0:
            case_1 = params.copy()
            case_2 = params.copy()
            case_3 = params.copy()

            case_3["a_attitude"] = 0.05
            case_3["b_attitude"] = 0.05
            case_3["confirmation_bias"] = 40

            case_2["a_attitude"] = 0.3
            case_2["b_attitude"] = 0.3  
            case_2["confirmation_bias"] = 18#THIS IS NEEDS TO SPLIT PARALLEL

            case_1["a_attitude"] = 2.0
            case_1["b_attitude"] = 2.0
            case_1["confirmation_bias"] = 10

            params_list = [case_1, case_2, case_3]
            print(params_list)

        elif GRAPH_TYPE == 3:
            params["network_structure"] = "small_world" #0.05
            params["K"] = 5
            params["prob_rewire"] = 0.1

            case_1 = params.copy()
            case_2 = params.copy()
            case_3 = params.copy()

            # np.asarray([0.2, 0.6, 1.0])
            case_1["homophily"] = 0.2#40
            case_2["homophily"] = 0.6#40
            case_3["homophily"] = 1.0#40

            params["network_structure"] = "barabasi_albert_graph" #0.05
            params["k_new_node"] = 3#Number of edges to attach from a new node to existing nodes
            case_4 = params.copy()
            case_5 = params.copy()
            case_6 = params.copy()

            # np.asarray([0.2, 0.6, 1.0])
            case_4["homophily"] = 0.2#40
            case_5["homophily"] = 0.6#40
            case_6["homophily"] = 1.0#40

            params_list = [case_1, case_2, case_3, case_4,case_5, case_6]
        else:
            params_list = produce_param_list(params, property_values_list, property_varied)

        data = parallel_run(params_list)  # better if a Multiple of 4

        ###WORKING

        if GRAPH_TYPE == 0:
            #FOR POLARISATION A,B PLOT - NEED TO SET self.b_attitude = parameters["a_attitude"] in NETWORK
            live_print_culture_timeseries(fileName, data, property_varied, title_list, nrows, ncols, dpi_save)
        elif GRAPH_TYPE == 1:
            ################
            #FOR ALPHA CHANGE PLOT
            live_print_culture_timeseries_with_weighting(fileName, data, property_varied, title_list, nrows, ncols, dpi_save, cmap_weighting)
        elif GRAPH_TYPE == 2:
            ###############################
            #FOR HOMOPHILY PLOT
            print_live_intial_culture_networks_and_culture_timeseries(fileName, data, dpi_save, property_values_list, property_varied_title, ncols, layout, norm_zero_one, cmap, node_size,round_dec)
        elif GRAPH_TYPE == 3:
            #Data_dict = {"small_world": data[:3], "barabasi_albert_graph": data[3:]}
            #live_print_culture_timeseries(fileName, data, property_varied, title_list, nrows, ncols, dpi_save)
            layout = ["circular","circular", "circular", "spring", "spring", "spring"]
            print_live_intial_culture_networks_and_culture_timeseries(fileName, data, dpi_save, property_values_list, property_varied_title, ncols, layout, norm_zero_one, cmap, node_size,round_dec)
        elif GRAPH_TYPE ==4:
            live_print_culture_timeseries(fileName, data, property_varied, title_list, nrows, ncols, dpi_save)
            plot_carbon_emissions_total_comparison(fileName, data, dpi_save, property_values_list, property_varied_title, round_dec)
        else:
            pass
        
        """Comment out those plots that you dont want to produce"""
        #live_print_culture_timeseries(fileName, data, property_varied, title_list, nrows, ncols, dpi_save)
        # plot_average_culture_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_carbon_emissions_total_comparison(fileName, data, dpi_save, property_values_list, property_varied, round_dec)
        # plot_weighting_matrix_convergence_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        # plot_average_culture_no_range_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        # plot_live_link_change_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        # plot_live_cum_link_change_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        # plot_live_link_change_per_agent_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_cum_link_change_per_agent_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #live_varaince_timeseries( fileName,data,property_varied,property_varied_title,property_values_list,dpi_save,)
        #live_multirun_diagram_mean_coefficient_variance(fileName,data,property_varied,property_values_list,property_varied_title,cmap,dpi_save,norm_zero_one,)
        # print_live_intial_culture_networks(fileName, data, dpi_save, property_values_list, property_varied, nrows, ncols , layout, norm_zero_one, cmap, node_size,round_dec)
        # prints_init_weighting_matrix(fileName, data, dpi_save,nrows, ncols, cmap_weighting,property_values_list, property_varied,round_dec)
        # prints_final_weighting_matrix(fileName, data, dpi_save,nrows, ncols, cmap_weighting,property_values_list, property_varied,round_dec)
        #live_print_culture_timeseries_with_weighting(fileName, data, property_varied, title_list, nrows, ncols, dpi_save, cmap_weighting)
        #print_live_intial_culture_networks_and_culture_timeseries(fileName, data, dpi_save, property_values_list, property_varied_title, ncols, layout, norm_zero_one, cmap, node_size,round_dec)
        # ani_a =  multi_animation_weighting(fileName,data, cmap_weighting,  interval, fps, round_dec, nrows, ncols)
        # ani_b = live_compare_animate_culture_network_and_weighting(fileName,data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,property_varied_title,property_values_list)
        # ani_c = live_compare_animate_weighting_matrix(fileName, data,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,property_varied_title,property_values_list)
        # ani_d = live_compare_animate_behaviour_matrix(fileName, data,  cmap, interval, fps, round_dec, nrows, ncols,property_varied_title,property_values_list)
        # ani_e = live_compare_plot_animate_behaviour_scatter(fileName,data,norm_zero_one, cmap, nrows, ncols,property_varied, property_values_list,interval, fps,round_dec)
    else:
        # AVERAGE OVER MULTIPLE RUNS
        """Set the number of stochastic repetitions by changing the number of entries in the seed list."""
        seed_list = [1, 2, 3, 4, 5]  # ie 5 reps per run!
        params["seed_list"] = seed_list
        average_reps = len(seed_list)

        fileName = "results/average_%s_variation_%s_%s_%s_%s_%s_%s_%s" % (
            property_varied,
            str(params["N"]),
            str(params["time_steps_max"]),
            str(params["K"]),
            str(param_min),
            str(param_max),
            str(reps),
            str(average_reps),
        )
        print("fileName: ", fileName)
        createFolder(fileName)

        params_list = produce_param_list(params, property_values_list, property_varied)
        (
            results_emissions,
            results_mean,
            results_var,
            results_coefficient_variance,
        ) = parallel_run_sa(params_list)

        """
        live_average_multirun_diagram_mean_coefficient_variance(
            fileName,
            results_mean,
            results_coefficient_variance,
            property_varied,
            property_values_list,
            property_varied_title,
            cmap_weighting,
            dpi_save,
            log_norm,
        )
        """
    plt.show()
