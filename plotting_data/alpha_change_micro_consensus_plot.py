# imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import get_cmap
from resources.utility import save_object,load_object, get_cluster_list
from resources.plot import (
    live_print_culture_timeseries_with_weighting,
    calc_num_clusters_set_bandwidth,
    plot_cluster_culture_time_series,
    plot_joint_cluster_micro,
    load_plot_alpha_group_single

)
from scipy.signal import argrelextrema

def calc_groups( Data,s, bandwidth) :

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

def main(
    fileName = "results/alpha_change_micro_consensus_single_13_43_21__08_01_2023",
    data_analysis_clusters = 0,
    alpha_change_plot = 0,
    micro_clusters_plot = 0,
    joint_cluster_micro = 1,
    dpi_save = 600,
    shuffle_colours = False,
    ) -> None: 

    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    ),
    norm_zero_one = Normalize(vmin=0, vmax=1),
    cmap_weighting = get_cmap("Reds"),
    cmap_multi = get_cmap("plasma"),
    
    data_list = load_object(fileName + "/Data", "data_list")
    property_varied = load_object(fileName + "/Data", "property_varied")
    title_list = load_object(fileName + "/Data", "title_list")
    
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


