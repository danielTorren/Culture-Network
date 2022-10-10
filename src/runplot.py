"""Runs a single simulation to produce data which is saved and plotted 
A module that use dictionary of data for the simulation run. The single shot simualtion is run
for a given intial set seed. The desired plots are then produced and saved.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
#imports
from logging import raiseExceptions
from run import run
from plot import (
    plot_culture_timeseries,
    animate_weighting_matrix,
    animate_behavioural_matrix,
    animate_culture_network,
    prints_behavioural_matrix,
    prints_culture_network,
    multi_animation,
    multi_animation_alt,
    multi_animation_scaled,
    plot_value_timeseries,
    plot_threshold_timeseries,
    plot_attitude_timeseries,
    standard_behaviour_timeseries_plot,
    plot_carbon_price_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_av_carbon_emissions_timeseries,
    prints_weighting_matrix,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_culture_timeseries,
    plot_beta_distributions,
    print_network_social_component_matrix,
    animate_network_social_component_matrix,
    animate_network_information_provision,
    print_network_information_provision,
    multi_animation_four,
    print_culture_histogram,
    animate_culture_network_and_weighting,
    plot_weighting_link_timeseries,
    plot_green_adoption_timeseries,
    prints_behaviour_timeseries_plot_colour_culture,
    live_plot_heterogenous_culture_momentum,
    Euclidean_cluster_plot,
    plot_k_cluster_scores,
    plot_behaviour_scatter,
    animate_behaviour_scatter,
)
from utility import loadData, get_run_properties, frame_distribution_prints,k_means_calc,loadObjects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize,SymLogNorm
from matplotlib.cm import get_cmap
import time
import numpy as np

#constants
params = {
    "total_time": 2000,#200,
    "delta_t": 1.0,#0.05,
    "compression_factor": 10,
    "save_data": True, 
    "alpha_change" : 1.0,
    "harsh_data": False,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.001,
    "phi_upper": 0.005,
    "N": 20,
    "M": 5,
    "K": 10,
    "prob_rewire": 0.2,#0.05,
    "set_seed": 1,
    "culture_momentum_real": 100,#5,
    "learning_error_scale": 0.02,
    "discount_factor": 0.8,
    "present_discount_factor": 0.99,
    "inverse_homophily": 0.2,#0.1,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : 1,
    "confirmation_bias": -100,
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
    params["alpha_attitude"] = 0.1
    params["beta_attitude"] = 0.1
    params["alpha_threshold"] = 1
    params["beta_threshold"] = 1


# SAVING DATA
params_name = [#THOSE USEd to create the save list?
    params["time_steps_max"],
    params["M"],
    params["N"],
    params["delta_t"],
    params["K"],
    params["prob_rewire"],
    params["set_seed"],
    params["learning_error_scale"],
    params["culture_momentum_real"],
]

# THINGS TO SAVE
data_save_behaviour_array_list = ["value", "attitude", "threshold"]
data_save_individual_list = ["culture", "carbon_emissions"]
data_save_network_list = [
    "time",
    "cultural_var",
    "total_carbon_emissions",
    "weighting_matrix_convergence",
    "average_culture",
    "min_culture",
    "max_culture",
    "green_adoption",
] 
data_save_network_array_list = [
    "weighting_matrix",
    "social_component_matrix",
]

to_save_list = [
    data_save_behaviour_array_list,
    data_save_individual_list,
    data_save_network_list,
    data_save_network_array_list,
]

# LOAD DATA
paramList = [
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
loadBooleanCSV = [
    "individual_culture",
    "individual_carbon_emissions",
    "network_total_carbon_emissions",
    "network_time",
    "network_cultural_var",
    "network_weighting_matrix_convergence",
    "network_average_culture",
    "network_min_culture",
    "network_max_culture",
    "network_green_adoption",
] 
loadBooleanArray = [
    "network_weighting_matrix",
    "network_social_component_matrix",
    "behaviour_value",
    "behaviour_threshold",
    "behaviour_attitude",
]


###PLOT STUFF
nrows_behave = 1
ncols_behave = params["M"]
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1)

#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)
log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct

#log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

alpha_quick, alpha_normal, alpha_lagard = 0.9,0.7,0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"

#print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1

min_val = 1e-3

bin_num = 1000
num_counts = 100000
bin_num_agents = int(round(params["N"]/10))
dpi_save = 2000

min_k,max_k = 2,10#N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

RUN = 1#False
LOAD_LIVE_DATA = 1
LOAD_STATIC_DATA = 1
PLOT = 1
cluster_plots = 0
SHOW_PLOT = 1


if __name__ == "__main__":

    if RUN == False:
        FILENAME = "results/_2000_8_100_1.0_10_0.7_1_0.02_100"
    else:
        # start_time = time.time()
        # print("start_time =", time.ctime(time.time()))
        ###RUN MODEL
        #print("start_time =", time.ctime(time.time()))
        FILENAME, social_network = run(params, to_save_list, params_name)
        #print("Final confirmation bias: ",social_network.confirmation_bias)
        # print ("RUN time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

    if PLOT:
        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        dataName = FILENAME + "/Data"

        if LOAD_LIVE_DATA:
            "LOAD LIVE OBJECT"
            live_network = loadObjects(dataName)

        if LOAD_STATIC_DATA:
            "LOAD DATA"
            Data = loadData(dataName, loadBooleanCSV, loadBooleanArray)
            Data = get_run_properties(Data, FILENAME, paramList)


        #####BODGES!!
        Data["network_time"] = np.asarray(Data["network_time"])[
            0
        ]  # for some reason pandas does weird shit
        Data["phi_list"] = np.linspace(params["phi_lower"], params["phi_upper"], num=params["M"])# ALSO DONE IN NETWORK BUT NEEDED FOR PLOTS AND HAVENT SAVED IT AS OF YET# ITS A PAIN TO GET IT IN

        frames_list = [int(round(x)) for x in np.linspace(0, len(Data["network_time"])-1 , num=frame_num + 1)]# -1 is so its within range as linspace is inclusive

        ###PLOTS
        #plot_beta_distributions(FILENAME,alpha_attitude,beta_attitude,alpha_threshold,beta_threshold,bin_num,num_counts,dpi_save,)
        plot_culture_timeseries(FILENAME, Data, dpi_save)
        #plot_green_adoption_timeseries(FILENAME, Data, dpi_save)
        #plot_value_timeseries(FILENAME,Data,nrows_behave, ncols_behave,dpi_save)
        #plot_threshold_timeseries(FILENAME,Data,nrows_behave, ncols_behave,dpi_save)
        plot_attitude_timeseries(FILENAME, Data, nrows_behave, ncols_behave, dpi_save)
        #plot_total_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        #plot_av_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        #plot_weighting_matrix_convergence_timeseries(FILENAME, Data, dpi_save)
        #plot_cultural_range_timeseries(FILENAME, Data, dpi_save)
        #plot_average_culture_timeseries(FILENAME,Data,dpi_save)
        #plot_weighting_link_timeseries(FILENAME, Data, "Link strength", dpi_save,min_val)
        #plot_behaviour_scatter(FILENAME,Data,"behaviour_attitude",dpi_save)

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


        if cluster_plots:
            k_clusters,win_score, scores = k_means_calc(Data,min_k,max_k,size_points)#CALCULATE THE OPTIMAL NUMBER OF CLUSTERS USING SILOUTTE SCORE, DOENST WORK FOR 1
            #k_clusters = 2 # UNCOMMENT TO SET K MANUALLY
            Euclidean_cluster_plot(FILENAME, Data, k_clusters,alpha_val,min_culture_distance, dpi_save)

        print(
            "PLOT time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )

    if SHOW_PLOT:
        plt.show()
