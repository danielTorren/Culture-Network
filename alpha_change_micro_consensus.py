# imports
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.cm import get_cmap
from resources.utility import createFolder,produce_name_datetime,save_object,load_object,calc_pos_clusters_set_bandwidth, get_cluster_list
from resources.run import parallel_run 
from resources.plot import (
    plot_alpha_group_multi,
    live_print_culture_timeseries_with_weighting,
    plot_cluster_culture_time_series_multi,
    calc_num_clusters_set_bandwidth,
    plot_cluster_culture_time_series,
)
from resources.multi_run_single_param import (
    produce_param_list,
)
from scipy.signal import argrelextrema

def calc_groups( Data,s, bandwidth):

    fig, ax = plt.subplots()

    culture_data = np.asarray([Data.agent_list[n].culture for n in range(Data.N)])

    kde, e = calc_num_clusters_set_bandwidth(culture_data,s,bandwidth)
    
    print("bandwidth used:",bandwidth)

    mi = argrelextrema(e, np.less)[0]#list of minimum values in the kde
    ma = argrelextrema(e, np.greater)[0]#list of minimum values in the kde
    
    clusters_index_lists = get_cluster_list(culture_data,s, Data.N, mi)

    time_vals_data = []
    for t in range(len(Data.history_time)):
        time_vals_data_row = []
        for i in range(len(clusters_index_lists)):
            #print("clusters_index_lists[i]",clusters_index_lists[i],len(clusters_index_lists[i]))
            sub_weighting_matrix = Data.history_weighting_matrix[t][clusters_index_lists[i]]
            #print("sub_weighting_matrix", sub_weighting_matrix, sub_weighting_matrix.shape)
            #i want a matrix that excludes all the values that arent from the indes in the clusters_index_lists[i]
            sub_sub_weighting_matrix = sub_weighting_matrix[:,clusters_index_lists[i]]
            #print("sub_sub_weighting_matrix", sub_sub_weighting_matrix, sub_sub_weighting_matrix.shape)

            mean_weighting_val = np.mean(sub_sub_weighting_matrix)
            #print("mean_value",mean_weighting_val)

            time_vals_data_row.append(mean_weighting_val)
            #quit()
        time_vals_data.append(time_vals_data_row)
    
    time_vals_data_array = np.asarray(time_vals_data)
    #print("time_vals_data_array", time_vals_data_array.shape)
    vals_time_data = time_vals_data_array.T
    #print("vals_time_data ",vals_time_data.shape)

    cluster_example_identity_list = s[ma]

    return clusters_index_lists,cluster_example_identity_list, vals_time_data

def load_plot_alpha_group_single(fileName, Data, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, auto_bandwidth, bandwidth,cmap, norm_zero_one,):

    fig, ax = plt.subplots()

    inverse_N_g_list = [1/len(i) for i in clusters_index_lists]

    colour_adjust = norm_zero_one(cluster_example_identity_list)
    ani_step_colours = cmap(colour_adjust)

    for i in range(len(clusters_index_lists)): 
        ax.plot(Data.history_time, vals_time_data[i], color = ani_step_colours[i], label = "Cluster %s" % (i + 1))#
        ax.axhline(y= inverse_N_g_list[i], color = ani_step_colours[i], linestyle = "--")

    #ax.set_title(title_list[z])
    ax.legend()

    ax.set_ylabel(r"Cluster center Identity, $I_{t,n}$")
    ax.set_xlabel(r"Time")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=ax
    )
    cbar.set_label(r"Identity, $I_{t,n}$")

    plotName = fileName + "/Prints"
    f = plotName + "/ plot_alpha_group_single_%s_%s" % (auto_bandwidth, bandwidth)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_joint_cluster_micro(fileName, Data, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, auto_bandwidth, bandwidth,cmap_multi, norm_zero_one,shuffle_colours):
    
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,6), constrained_layout=True)

    ###################################################
    
    #colour_adjust = norm_zero_one(cluster_example_identity_list)
    #ani_step_colours = cmap(colour_adjust)

    cmap = get_cmap(name='viridis', lut = len(cluster_example_identity_list))
    ani_step_colours = [cmap(i) for i in range(len(cluster_example_identity_list))] 

    if shuffle_colours:
        np.random.shuffle(ani_step_colours)
    #else:
        #cbar = fig.colorbar(
        #    plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes[0]
        #)
        #cbar.set_label(r"Cluster center Identity, $I_{t,n}$")
    #print("ani_step_colours",ani_step_colours)

    colours_dict = {}#It cant be a list as you need to do it out of order
    for i in range(len(clusters_index_lists)):#i is the list of index in that cluster
        for j in clusters_index_lists[i]:#j is an index in that cluster
            #print(i,j)
            colours_dict["%s" % (j)] = ani_step_colours[i]
        
    #print("colours_dict",colours_dict)

    for v in range(len(Data.agent_list)):
        axes[0].plot(np.asarray(Data.history_time), np.asarray(Data.agent_list[v].history_culture), color = colours_dict["%s" % (v)])

    axes[0].set_ylabel(r"Identity, $I_{t,n}$")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel(r"Time")

    ##################################################

    inverse_N_g_list = [1/len(i) for i in clusters_index_lists]

    #colour_adjust = norm_zero_one(cluster_example_identity_list)
    #ani_step_colours = cmap(colour_adjust)

    for i in range(len(clusters_index_lists)): 
        axes[1].plot(Data.history_time, vals_time_data[i], color = ani_step_colours[i])#, label = "Cluster %s" % (i + 1)
        axes[1].axhline(y= inverse_N_g_list[i], color = ani_step_colours[i], linestyle = "--")

    #ax.set_title(title_list[z])
    #axes[1].legend()

    axes[1].set_ylabel(r"Mean cluster weighting")
    axes[1].set_xlabel(r"Time")

    #cbar = fig.colorbar(
    #    plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one), ax=axes[1]
    #)
    #cbar.set_label(r"Identity, $I_{t,n}$")

    plotName = fileName + "/Prints"
    f = plotName + "/plot_joint_cluster_micro_%s_%s" % (auto_bandwidth, bandwidth)
    fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

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

shuffle_colours = False

if __name__ == "__main__":

    RUN = 0

    data_analysis_clusters = 0

    alpha_change_plot = 0
    micro_clusters_plot = 0
    joint_cluster_micro = 1

    #fileName_clustering_colours = "results/one_param_sweep_single_19_02_23__14_12_2022"
    #fileName_alpha = "results/alpha_change_variation_50_2000_4_0.0_1.0_3"#this is the one used for the graph would be good to give the clustering for the last one!

    # $\theta = 40$, $a,b=0.5$, $N = 150$, $M = 3$, $K = 4$

    fileName = "results/alpha_change_micro_consensus_single_13_43_21__08_01_2023"

    if RUN:
        ################
        #FOR ALPHA CHANGE PLOT
        property_varied = "alpha_change"
        property_varied_title = "alpha_change"
        param_min = 0.0
        param_max = 1.0  # 50.0
        title_list = [r"Static uniform $\alpha_{n,k}$", r"Static culturally determined $\alpha_{n,k}$", r"Dynamic culturally determined $\alpha_{n,k}$"]
        property_values_list = np.asarray([0.0, 0.5, 1.0])# FOR ALPHA

        f = open("constants/base_params.json")
        base_params = json.load(f)
        base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

        root = "alpha_change_micro_consensus_single"
        fileName = produce_name_datetime(root)
        print("fileName: ", fileName)

        dataName = fileName + "/Data"

        params_list = produce_param_list(base_params, property_values_list, property_varied)

        data_list = parallel_run(params_list)  # better if a Multiple of 4
        createFolder(fileName)

        save_object(data_list, fileName + "/Data", "data_list")
        save_object(base_params, fileName + "/Data", "base_params")
        save_object(property_varied, fileName + "/Data", "property_varied")
        save_object(property_varied_title, fileName + "/Data", "property_varied_title")
        save_object(param_min, fileName + "/Data", "param_min")
        save_object(param_max, fileName + "/Data", "param_max")
        save_object(title_list, fileName + "/Data", "title_list")
        save_object(property_values_list, fileName + "/Data", "property_values_list")
    else:
        data_list = load_object(fileName + "/Data", "data_list")
        params = load_object(fileName + "/Data", "base_params")
        property_varied = load_object(fileName + "/Data", "property_varied")
        property_varied_title = load_object(fileName + "/Data", "property_varied_title")
        param_min = load_object(fileName + "/Data", "param_min")
        param_max = load_object(fileName + "/Data", "param_max")
        title_list = load_object(fileName + "/Data", "title_list")
        property_values_list = load_object(fileName + "/Data", "property_values_list")
    
    if alpha_change_plot:
        ################
        #FOR ALPHA CHANGE PLOT
        nrows, ncols = 1,3
        live_print_culture_timeseries_with_weighting(fileName, data_list, property_varied, title_list, nrows, ncols, dpi_save, cmap_weighting)
    
    if micro_clusters_plot:

        data_list_reduc = data_list[2]

        if data_analysis_clusters:
            no_samples = 10000
            s = np.linspace(0, 1,no_samples)
            bandwidth = 0.01

            auto_bandwidth = False

            clusters_index_lists, cluster_example_identity_list, vals_time_data = calc_groups(data_list_reduc,s, bandwidth)#set it using input value
            
            save_object(clusters_index_lists, fileName + "/Data", "clusters_index_lists")
            save_object(cluster_example_identity_list, fileName + "/Data", "cluster_example_identity_list")
            save_object(vals_time_data, fileName + "/Data", "vals_time_data")
            save_object(bandwidth, fileName + "/Data", "bandwidth")
            save_object(auto_bandwidth, fileName + "/Data", "auto_bandwidth")
        else:
            dataName = fileName + "/Data"
            clusters_index_lists = load_object(dataName,"clusters_index_lists")
            cluster_example_identity_list = load_object(dataName,"cluster_example_identity_list")
            vals_time_data = load_object(dataName,"vals_time_data")
            bandwidth = load_object( fileName + "/Data", "bandwidth")
            auto_bandwidth = load_object( fileName + "/Data", "auto_bandwidth")
        
        if joint_cluster_micro:
            plot_joint_cluster_micro(fileName, data_list_reduc, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, auto_bandwidth, bandwidth,cmap_multi, norm_zero_one,shuffle_colours)
        else:
            load_plot_alpha_group_single(fileName, data_list_reduc, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, auto_bandwidth, bandwidth,cmap_multi, norm_zero_one,)
            plot_cluster_culture_time_series(fileName, data_list_reduc, dpi_save,clusters_index_lists,cluster_example_identity_list, cmap,norm_zero_one, shuffle_colours)

    plt.show()


