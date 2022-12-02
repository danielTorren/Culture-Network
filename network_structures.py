"""Vary network structures

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 01/11/2022
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (LinearSegmentedColormap, Normalize)
from resources.run import (generate_data)
from resources.utility import (
    createFolder,
    produce_name_datetime,
)
from resources.plot import prod_pos
import numpy as np
import networkx as nx
import collections

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

params = {
    "total_time": 3000,
    "delta_t": 1.0,
    "compression_factor": 10,
    "save_data": 1, 
    "alpha_change" : 1.0,
    "degroot_aggregation": 1,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.01,
    "phi_upper": 0.05,
    "N": 200,
    "M": 3,
    "K": 5,
    "prob_rewire": 0.1,
    "set_seed": 1,
    "seed_list": [1,2,3,4,5],
    "culture_momentum_real": 100,
    "learning_error_scale": 0.02,
    "discount_factor": 0.95,
    "homophily": 1,
    "homophilly_rate" : 1,
    "confirmation_bias": 10,
    "a_attitude": 0.5,
    "b_attitude": 0.5,
    "a_threshold": 1,
    "b_threshold": 1,
    "action_observation_I": 0.0,
    "action_observation_S": 0.0,
    "green_N": 0,
    "network_structure": "small_world"
}

def live_print_culture_timeseries_varynetwork_structure(
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    y_title = r"Identity, $I_{t,n}$"

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())
    
    structure_list_title = ["Watts-Strogatz small world", "Barabasi-Albert scale free"]
    
    for i, ax in enumerate(axes.flat):
        for v in structure_data[i].agent_list:
            ax.plot(
                np.asarray(structure_data[i].history_time), np.asarray(v.history_culture)
            )

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(structure_list_title[i])
        #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/live_print_culture_timeseries_varynetwork_structure_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def draw_networks(
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
    norm_zero_one,
    cmap,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())

    layout_list = ["circular", "spring"]
    structure_list_title = ["Watts-Strogatz small world", "Barabasi-Albert scale free"]
    for i, ax in enumerate(axes.flat):
        G = structure_data[i].network
        individual_culture = [x.history_culture[0] for x in structure_data[i].agent_list]#get intial culture
        colour_adjust = norm_zero_one(individual_culture)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            ax=ax,
            node_color=ani_step_colours,
            node_size=50,
            edgecolors="black",
            pos=prod_pos(layout_list[i], G),
        )

        #ax.set_xlabel(r"Time")
        #ax.set_ylabel(r"%s" % y_title)
        ax.set_title(structure_list_title[i])
        #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/draw_networks_network_structure_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def degree_distribution(   
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())

    for i, ax in enumerate(axes.flat):
        G = structure_data[i].network
        
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
        # print "Degree sequence", degree_sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        ax.bar(deg, cnt, width=0.80, color='b')

        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)

        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.set_title(structure_list[i])
        #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/degree_distribution_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def plot_average_culture_timeseries_std(    
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    y_title = "Identity"

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())

    for i, ax in enumerate(axes.flat):
        ax.plot(np.asarray(structure_data[i].history_time), np.asarray(structure_data[i].history_average_culture))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(structure_list[i])

        min_values = np.asarray(structure_data[i].history_average_culture) - np.asarray(structure_data[i].history_std_culture)
        max_values = np.asarray(structure_data[i].history_average_culture) + np.asarray(structure_data[i].history_std_culture)

        ax.fill_between(np.asarray(structure_data[i].history_time), min_values, max_values, alpha=0.5, linewidth=0)

    plotName = fileName + "/Prints"
    f = plotName + "/plot_average_culture_timeseries_std_network_structure_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

def identity_overlap(
    fileName,
    Data_dict,
    nrows,
    ncols,
    dpi_save,
):

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True
    )

    y_title = "Identity Overlap"

    structure_list = list(Data_dict.keys())
    structure_data = list(Data_dict.values())

    for i, ax in enumerate(axes.flat):

        matrix_data = np.asarray(structure_data[i].history_total_identity_differences)
        matrix_data_timeseries = matrix_data.transpose()
        for v in matrix_data_timeseries:
            ax.plot(np.asarray(structure_data[i].history_time), v)

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(structure_list[i])
        #ax.set_ylim(0, 1)

    plotName = fileName + "/Prints"
    f = plotName + "/identity_overlap_%s" % (
        len(structure_list),
    )
    #fig.savefig(f + ".eps", dpi=dpi_save, format="eps")
    fig.savefig(f + ".png", dpi=dpi_save, format="png")

if __name__ == "__main__":

    norm_zero_one = Normalize(vmin=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1
    )

    nrows = 1
    ncols = 2
    dpi_save = 1200
    root = "network_structure"
    fileName = produce_name_datetime(root)

    createFolder(fileName)

    params["time_steps_max"] = int(params["total_time"] / params["delta_t"])

    #run for different network strctures
    Data_dict = {}
    #small_world
    params["network_structure"] = "small_world" # https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html#networkx.generators.random_graphs.watts_strogatz_graph
    params["K"] = 5
    params["prob_rewire"] = 0.1
    Data_dict["small_world"] = generate_data(params)  # run the simulation
    
    """
    #erdos_renyi_graph
    params["network_structure"] = "erdos_renyi_graph" # https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html#networkx.generators.random_graphs.erdos_renyi_graph
    params["prob_edge"] = 0.1
    Data_dict["erdos_renyi_graph"] = generate_data(params)  # run the simulation
    """

    #barabasi_albert_graph - I believe this is scale free or power law graph
    params["network_structure"] = "barabasi_albert_graph" # https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html#networkx.generators.random_graphs.barabasi_albert_graph
    params["k_new_node"] = 3#Number of edges to attach from a new node to existing nodes
    Data_dict["barabasi_albert_graph"] = generate_data(params)

    for key,value in Data_dict.items():
        print("Density: ",key,value.network_density)
        #print("history_std_culture", key, value.history_std_culture)

    live_print_culture_timeseries_varynetwork_structure(fileName,Data_dict,nrows,ncols,dpi_save)
    draw_networks(fileName,Data_dict,nrows,ncols,dpi_save,norm_zero_one,cmap)
    #degree_distribution(fileName,Data_dict,nrows,ncols,dpi_save)
    #plot_average_culture_timeseries_std(fileName,Data_dict,nrows,ncols,dpi_save)
    #identity_overlap(fileName,Data_dict,nrows,ncols,dpi_save)

    plt.show()
