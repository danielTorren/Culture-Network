"""Conduct analysis and plots for varying the frequency of identity updating"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import  Normalize
from matplotlib.cm import get_cmap
from package.resources.utility import save_object,load_object, get_cluster_list,calc_num_clusters_set_bandwidth
from package.resources.plot import (
    live_print_culture_timeseries_with_weighting,
    plot_joint_cluster_micro,

)
from scipy.signal import argrelextrema

def calc_groups( Data,s, bandwidth) :

    culture_data = np.asarray([Data.agent_list[n].culture for n in range(Data.N)])

    kde, e = calc_num_clusters_set_bandwidth(culture_data,s,bandwidth)
    

    mi = argrelextrema(e, np.less)[0]#list of minimum values in the kde
    ma = argrelextrema(e, np.greater)[0]#list of minimum values in the kde
    
    clusters_index_lists = get_cluster_list(culture_data,s, Data.N, mi)

    time_vals_data = []
    for t in range(len(Data.history_time)):
        time_vals_data_row = []
        for i in range(len(clusters_index_lists)):
            sub_weighting_matrix = Data.history_weighting_matrix[t][clusters_index_lists[i]]
            sub_sub_weighting_matrix = sub_weighting_matrix[:,clusters_index_lists[i]]
            mean_weighting_val = np.mean(sub_sub_weighting_matrix)
            time_vals_data_row.append(mean_weighting_val)
        time_vals_data.append(time_vals_data_row)
    
    time_vals_data_array = np.asarray(time_vals_data)
    vals_time_data = time_vals_data_array.T

    cluster_example_identity_list = s[ma]

    return clusters_index_lists,cluster_example_identity_list, vals_time_data

def main(
    fileName = "results/alpha_change_micro_consensus_single_13_43_21__08_01_2023",
    data_analysis_clusters = 0,
    alpha_change_plot = 0,
    micro_clusters_plot = 0,
    dpi_save = 1200,
    shuffle_colours = False,
    bandwidth = 0.01
    ) -> None: 

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap_weighting = get_cmap("Reds")
    cmap_multi = get_cmap("plasma")
    
    data_list = load_object(fileName + "/Data", "data_list")
    property_varied = load_object(fileName + "/Data", "property_varied")
    title_list = load_object(fileName + "/Data", "title_list")
    
    if alpha_change_plot:
        ################
        #FOR ALPHA CHANGE PLOT
        live_print_culture_timeseries_with_weighting(fileName, data_list, property_varied, title_list, dpi_save, cmap_weighting)
    
    if micro_clusters_plot:

        data_list_reduc = data_list[2]

        if data_analysis_clusters:
            no_samples = 10000
            s = np.linspace(0, 1,no_samples)
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
        
        plot_joint_cluster_micro(fileName, data_list_reduc, clusters_index_lists,cluster_example_identity_list, vals_time_data, dpi_save, auto_bandwidth, bandwidth,cmap_multi, norm_zero_one,shuffle_colours)

    plt.show()


