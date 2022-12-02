"""Runs a single simulation to produce data which is saved and plotted 
A module that use dictionary of data for the simulation run. The single shot simualtion is run
for a given intial set seed. The desired plots are then produced and saved.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm
from matplotlib.cm import get_cmap
import time
import json
import numpy as np
from resources.run import generate_data
from resources.utility import (
    createFolder, 
    save_object, 
    load_object,
    produceName,
    produce_name_datetime
)
from resources.plot import (
    plot_culture_timeseries,
    plot_cum_link_change_per_agent,
    # animate_weighting_matrix,
    # animate_behavioural_matrix,
    # animate_culture_network,
    # prints_behavioural_matrix,
    # prints_culture_network,
    # multi_animation,
    # multi_animation_alt,
    # multi_animation_scaled,
    plot_value_timeseries,
    plot_threshold_timeseries,
    plot_attitude_timeseries,
    # standard_behaviour_timeseries_plot,
    # plot_carbon_price_timeseries,
    plot_total_carbon_emissions_timeseries,
    # plot_av_carbon_emissions_timeseries,
    # prints_weighting_matrix,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_culture_timeseries,
    # plot_beta_distributions,
    # print_network_social_component_matrix,
    # animate_network_social_component_matrix,
    # animate_network_information_provision,
    # print_network_information_provision,
    # multi_animation_four,
    # print_culture_histogram,
    # animate_culture_network_and_weighting,
    # plot_weighting_link_timeseries,
    plot_green_adoption_timeseries,
    # prints_behaviour_timeseries_plot_colour_culture,
    # live_plot_heterogenous_culture_momentum,
    # plot_behaviour_scatter,
    # animate_behaviour_scatter,
    weighting_histogram,
    live_animate_weighting_matrix,
    weighting_histogram_time,
    cluster_estimation_plot,
)

# FOR FILENAME
params_name = [  # THOSE USEd to create the save list?
    "time_steps_max",
    "M",
    "N",
    "delta_t",
    "K",
    "prob_rewire",
    "set_seed",
    "learning_error_scale",
    "culture_momentum_real",
]

###PLOT STUFF
node_size = 50
cmap = LinearSegmentedColormap.from_list(
    "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
)

# norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one = Normalize(vmin=0, vmax=1)
log_norm = SymLogNorm(
    linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10
)  # this works at least its correct

# log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

alpha_quick, alpha_normal, alpha_lagard = 0.9, 0.7, 0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"

# print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1


dpi_save = 2000

min_k, max_k = (
    2,
    10,
)  # N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

bin_num = 20

RUN = 1
PLOT = 1
SHOW_PLOT = 1

if __name__ == "__main__":
    if RUN == False:
        FILENAME = "results/single_shot_10_48_40__02_12_2022"
    else:
        f = open("constants/base_params.json")
        base_params = json.load(f)
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        #FILENAME = produceName(params, params_name)
        root = "single_shot"
        FILENAME = produce_name_datetime(root)
        print("FILENAME:", FILENAME)

        Data = generate_data(base_params)  # run the simulation

        createFolder(FILENAME)
        save_object(Data, FILENAME + "/Data", "social_network")
        save_object(base_params, FILENAME + "/Data", "base_params")

    if PLOT:
        start_time = time.time()
        print("start_time =", time.ctime(time.time()))
        dataName = FILENAME + "/Data"
        Data = load_object(dataName, "social_network")

        no_samples = 10000
        s = np.linspace(0, 1,no_samples)
        bandwidth = 0.001

        ###PLOTS
        #bandwidth_list = [0.05]
        #cluster_estimation(Data,bandwidth_list)
        cluster_estimation_plot(Data,s,bandwidth)
        #plot_culture_timeseries(FILENAME, Data, dpi_save)
        #weighting_histogram(FILENAME, Data, dpi_save,bin_num)
        #weighting_histogram_time(FILENAME, Data, dpi_save,bin_num,300)
        #plot_green_adoption_timeseries(FILENAME, Data, dpi_save)
        #plot_total_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        #plot_weighting_matrix_convergence_timeseries(FILENAME, Data, dpi_save)
        # plot_cultural_range_timeseries(FILENAME, Data, dpi_save)
        # plot_average_culture_timeseries(FILENAME,Data,dpi_save)
        # plot_cum_link_change_per_agent(FILENAME,Data,dpi_save)

        #plot_value_timeseries(FILENAME,Data,dpi_save)
        #plot_threshold_timeseries(FILENAME,Data,dpi_save)
        #plot_attitude_timeseries(FILENAME,Data,dpi_save)

        #live_animate_weighting_matrix(FILENAME, Data,  cmap_weighting, interval, fps, round_dec)

        """
        #####BROKEN ATM
        ###PRINTS
        #prints_weighting_matrix(FILENAME,Data,cmap_weighting,nrows,ncols,frames_list,round_dec,dpi_save)
        #prints_behavioural_matrix(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec,dpi_save)
        #prints_culture_network(FILENAME,Data,layout,cmap,node_size,nrows,ncols,norm_zero_one,frames_list,round_dec,dpi_save)
        #print_network_social_component_matrix(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec,dpi_save)
        #print_culture_histogram(FILENAME, Data, "individual_culture", nrows, ncols, frames_list,round_dec,dpi_save, bin_num_agents)
        #prints_behaviour_timeseries_plot_colour_culture(FILENAME, Data, "behaviour_attitude", "attitudeiveness", nrows_behave, ncols_behave, dpi_save,cmap,norm_zero_one)
        ###ANIMATIONS
        #ani_b = animate_network_social_component_matrix(FILENAME,Data,interval,fps,round_dec,cmap,norm_zero_one)
        #ani_c = animate_weighting_matrix(FILENAME,Data,interval,fps,round_dec,cmap_weighting)
        #ani_d = animate_behavioural_matrix(FILENAME,Data,interval,fps,cmap,round_dec)
        #ani_e = animate_culture_network(FILENAME,Data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec)
        #ani_f = animate_culture_network_and_weighting(FILENAME,Data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge)
        #Shows the 2D movement of attitudes and their culture, equivalent to prints_behaviour_timeseries_plot_colour_culture
        #ani_l = animate_behaviour_scatter(FILENAME,Data,"behaviour_attitude",norm_zero_one, cmap,interval,fps,round_dec)
        """
        print(
            "PLOT time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    if SHOW_PLOT:
        plt.show()
