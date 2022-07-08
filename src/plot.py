import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import numpy as np
from utility import frame_distribution, frame_distribution_prints
from pandas import DataFrame
from matplotlib.colors import LinearSegmentedColormap,SymLogNorm
from typing import Union
from networkx import Graph
from network import Network
###DEFINE PLOTS


def prints_behaviour_timeseries_plot(
    FILENAME: str, Data: DataFrame, property:str, y_title:str, nrows:int, ncols:int, dpi_save:int
):
    PropertyData = Data[property].transpose()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print("property = ", property)

    for i, ax in enumerate(axes.flat):
        for j in range(int(Data["N"])):
            ax.plot(Data["network_time"], PropertyData[i][j])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(r"Trait %s" % (i + 1))  # avoid 0 in the title
        ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    plt.tight_layout()

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_prints_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def standard_behaviour_timeseries_plot(FILENAME: str, Data: DataFrame, property:str, y_title:str, dpi_save:int):
    PropertyData = Data[property].transpose()

    fig, ax = plt.subplots()
    for i in range(int(Data["N"])):
        for v in range(int(Data["M"])):
            ax.plot(Data["network_time"], PropertyData[i][v])
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def plot_value_timeseries(FILENAME: str, Data: DataFrame, nrows:int, ncols:int, dpi_save:int):
    prints_behaviour_timeseries_plot(
        FILENAME, Data, "behaviour_value", "Trait Value", nrows, ncols, dpi_save
    )


def plot_threshold_timeseries(FILENAME: str, Data: DataFrame, nrows:int, ncols:int, dpi_save:int):
    prints_behaviour_timeseries_plot(
        FILENAME, Data, "behaviour_threshold", "Threshold", nrows, ncols, dpi_save
    )


def plot_attract_timeseries(FILENAME: str, Data: DataFrame, nrows:int, ncols:int, dpi_save:int):
    
    #print(Data["behaviour_attract"],np.shape(Data["behaviour_attract"]))


    prints_behaviour_timeseries_plot(
        FILENAME, Data, "behaviour_attract", "Attractiveness", nrows, ncols, dpi_save
    )


def plot_av_carbon_emissions_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Carbon Emissions Per Individual"
    property = "individual_carbon_emissions"

    fig, ax = plt.subplots()
    av_network_total_carbon_emissions = [
        x / Data["N"] for x in np.asarray(Data["network_total_carbon_emissions"])[0]
    ]
    ax.plot(Data["network_time"], av_network_total_carbon_emissions, "k-")
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    data = np.asarray(Data["individual_carbon_emissions"])  # bodge

    for i in range(int(int(Data["N"]))):
        ax.plot(Data["network_time"], data[i])

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)

def plot_network_timeseries(FILENAME: str, Data: DataFrame,y_title:str, property:str,  dpi_save:int):

    fig, ax = plt.subplots()
    data = np.asarray(Data[property])[0]  # bodge
    ax.plot(Data["network_time"], data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def plot_carbon_price_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Carbon Price"
    property = "network_carbon_price"

    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_cultural_range_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Cultural Range"
    property = "network_cultural_var"
    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_weighting_matrix_convergence_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Change in Agent Link Strength"
    property = "network_weighting_matrix_convergence"
    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_total_carbon_emissions_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Carbon Emissions"
    property = "network_total_carbon_emissions"
    plot_network_timeseries(FILENAME, Data, y_title, property, dpi_save)


def plot_average_culture_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):
    y_title = "Average Culture"
    property = "network_average_culture"

    fig, ax = plt.subplots()
    data = np.asarray(Data[property])[0]  # bodge
    culture_min = np.asarray(Data["network_min_culture"])[0]  # bodge
    culture_max = np.asarray(Data["network_max_culture"])[0]  # bodge
    ax.plot(Data["network_time"], data)
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.fill_between(
        Data["network_time"], culture_min, culture_max, alpha=0.5, linewidth=0
    )

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def plot_culture_timeseries(FILENAME: str, Data: DataFrame, dpi_save:int):

    ##plot cultural evolution of agents
    fig, ax = plt.subplots()
    ax.set_xlabel("Time")
    ax.set_ylabel("Culture")

    data = np.asarray(Data["individual_culture"])  # bodge

    for i in range(int(int(Data["N"]))):
        # print(Data["individual_culture"][i])
        ax.plot(Data["network_time"], data[i])
    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + "cultural_evolution.png"
    fig.savefig(f, dpi=dpi_save)

# make animate_network_social_component_matrix
def animate_network_social_component_matrix(FILENAME: str, Data: DataFrame, interval:int, fps:int, round_dec:int, cmap: Union[LinearSegmentedColormap,str]):
    
    def update(i):
        M = Data["network_social_component_matrix"][i]
        # print("next frame!",M)        
        matrice.set_array(M)
        # Set the title        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots()

    matrice = ax.matshow(Data["network_social_component_matrix"][0], cmap=cmap, aspect="auto")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Behaviour")
    #plt.colorbar(matrice)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Social Learning")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(Data["time_steps_max"]),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "network_social_component_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

def animate_network_information_provision(FILENAME: str, Data: DataFrame, interval:int, fps:int, round_dec:int, cmap: Union[LinearSegmentedColormap,str]):

    def update(i):
        M = Data["behaviour_information_provision"][i]
        #print("next frame!",M, np.shape(M))  
      
        matrice.set_array(M)
        # Set the title        
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots()

    matrice = ax.matshow(Data["behaviour_information_provision"][0], cmap=cmap, aspect="auto")
    ax.set_xlabel("Behaviour")
    ax.set_ylabel("Agent")
    #plt.colorbar(matrice)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=Data["behaviour_information_provision"].max())),
        ax=ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Information Provision")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(Data["time_steps_max"]),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "behaviour_information_provision_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

# make matrix animation
def animate_weighting_matrix(FILENAME: str, Data: DataFrame, interval:int, fps:int, round_dec:int, cmap_weighting: Union[LinearSegmentedColormap,str]):
    def update(i):
        M = Data["network_weighting_matrix"][i]
        # print("next frame!",M)        
        matrice.set_array(M)
        # Set the title
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )
        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel("Agent")
    ax.set_ylabel("Agent")
    matrice = ax.matshow(Data["network_weighting_matrix"][0], cmap=cmap_weighting)
    plt.colorbar(matrice)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(Data["time_steps_max"]),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


# make behaviour evolution plot
def animate_behavioural_matrix(
    FILENAME: str, Data: DataFrame, interval:int, fps:int, cmap_behaviour: Union[LinearSegmentedColormap,str], round_dec:int
):
    def update(i):
        M = Data["behaviour_value"][i]
        # print("next frame!",M)
        matrice.set_array(M)

        # Set the title
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][i], round_dec))
            )

        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel("Behaviour")
    ax.set_ylabel("Agent")

    matrice = ax.matshow(Data["behaviour_value"][0], cmap=cmap_behaviour, aspect="auto")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Behavioural Value")

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(Data["time_steps_max"]),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "behavioural_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


def prod_pos(layout_type:str, network:Graph) -> Graph:

    if layout_type == "circular":
        pos_culture_network = nx.circular_layout(network)
    elif layout_type == "spring":
        pos_culture_network = nx.spring_layout(network)
    elif layout_type == "kamada_kawai":
        pos_culture_network = nx.kamada_kawai_layout(network)
    elif layout_type == "planar":
        pos_culture_network = nx.planar_layout(network)
    else:
        raise Exception("Invalid layout given")

    return pos_culture_network


# animation of changing culture
def animate_culture_network(
    FILENAME: str, Data: DataFrame, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_neg_pos: SymLogNorm, round_dec:int
):
    def update(i, G, pos, ax, cmap_culture):

        ax.clear()
        # print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
        colour_adjust = norm_neg_pos(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos,
            node_size=node_size,
            edgecolors="black",
        )

        # Set the title
        ax.set_title("Time= {}".format(round(Data["network_time"][i], round_dec)))

    # Build plot
    fig, ax = plt.subplots()
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Culture")

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(Data["time_steps_max"]),
        fargs=(G, pos_culture_network, ax, cmap_culture),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "cultural_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


def prints_weighting_matrix(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data["network_weighting_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data["network_weighting_matrix"][frames_list[i]],
            cmap=cmap_behaviour,
            aspect="auto",
        )
        # Set the title
        # print("Time= {}".format(round(Data["network_time"][frames_list[i]],round_dec)))
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")
    plt.tight_layout()

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),
        cax=cbar_ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Weighting matrix")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_weighting_matrix.png"
    fig.savefig(f, dpi=dpi_save)


def prints_behavioural_matrix(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        ax.matshow(
            Data["behaviour_value"][frames_list[i]], cmap=cmap_behaviour, aspect="auto"
        )
        # Set the title
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Agent")
    plt.tight_layout()

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        cax=cbar_ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Behavioural Value")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_behavioural_matrix.png"
    fig.savefig(f, dpi=dpi_save)

def print_network_social_component_matrix(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data[ "network_social_component_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data["network_social_component_matrix"][frames_list[i]],
            cmap=cmap_behaviour,
            aspect="auto",
        )
        # Set the title
        # print("Time= {}".format(round(Data["network_time"][frames_list[i]],round_dec)))
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Agent")
    plt.tight_layout()

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        cax=cbar_ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Social Learning")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_network_social_component_matrix.png"
    fig.savefig(f, dpi=dpi_save)

def print_network_information_provision(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], nrows:int, ncols:int, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data[ "network_social_component_matrix"]),frames_list)
    #
    #print(Data["behaviour_information_provision"].max())
    
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data["behaviour_information_provision"][frames_list[i]],
            cmap=cmap_behaviour,
            aspect="auto",
        )
        # Set the title
        # print("Time= {}".format(round(Data["network_time"][frames_list[i]],round_dec)))
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Agent")
    plt.tight_layout()

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),
        cax=cbar_ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Information Provision")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_behaviour_information_provision_matrix.png"
    fig.savefig(f, dpi=dpi_save)


def prints_culture_network(
    FILENAME: str, Data: DataFrame,layout:str, cmap_culture: LinearSegmentedColormap,node_size:int, nrows:int, ncols:int, norm_neg_pos: SymLogNorm, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    for i, ax in enumerate(axes.flat):
        # print(i,ax)
        ax.set_title(
            "Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec))
        )

        colour_adjust = norm_neg_pos(Data["individual_culture"][frames_list[i]])
        #colour_adjust = (Data["individual_culture"][frames_list[i]] + 1)/2
        #colour_adjust = Data["individual_culture"][frames_list[i]]
        #print("colour_adjust", colour_adjust)
        ani_step_colours = cmap_culture(colour_adjust)
        #print("ani_step_colours",ani_step_colours)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

    plt.tight_layout()

    # print("cmap_culture", cmap_culture)

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-1, vmax=1)), cax=cbar_ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Culture")

    f = FILENAME + "/Prints/prints_culture_network.png"
    fig.savefig(f, dpi=dpi_save)


def multi_animation(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], cmap_culture: Union[LinearSegmentedColormap,str], layout: str, node_size:int,  interval:int,
    fps:int, norm_neg_pos: SymLogNorm,
):

    ####ACUTAL

    fig = plt.figure(figsize=[7, 7])  # figsize = [8,5]
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Culture")
    data = np.asarray(Data["individual_culture"])  # bodge
    for i in range(int(Data["N"])):
        ax3.plot(Data["network_time"], data[i])

    lines = [-1, -4 / 6, -2 / 6, 0, 2 / 6, 4 / 6, 1]

    for i in lines:
        ax3.axhline(y=i, color="b", linestyle="--", alpha=0.3)

    ax3.grid()
    time_line = ax3.axvline(x=0.0, linewidth=2, color="r")

    ax1.set_xlabel("Behaviour")
    ax1.set_ylabel("Agent")

    ####CULTURE ANIMATION
    def update(i):
        ax2.clear()

        colour_adjust = norm_neg_pos(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax2,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

        M = Data["behaviour_value"][i]
        # print("next frame!",M)
        matrice.set_array(M)

        time_line.set_xdata(Data["network_time"][i])

        return matrice, time_line

    cbar_behave = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax1,
    )  # This does a mapabble on the fly i think, not sure
    cbar_behave.set_label("Behavioural Value")

    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_neg_pos), ax=ax2
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Culture")

    matrice = ax1.matshow(
        Data["behaviour_value"][0], cmap=cmap_behaviour, aspect="auto"
    )

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(Data["network_time"]),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


def multi_animation_alt(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], cmap_culture: Union[LinearSegmentedColormap,str], layout: str, node_size:int,  interval:int,
    fps:int, norm_neg_pos: SymLogNorm,
):

    ####ACUTAL

    fig = plt.figure(figsize=[7, 7])  # figsize = [8,5]
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Culture")
    data = np.asarray(Data["individual_culture"])  # bodge
    lines = [-1, -4 / 6, -2 / 6, 0, 2 / 6, 4 / 6, 1]

    for i in lines:
        ax3.axhline(y=i, color="b", linestyle="--", alpha=0.3)

    ax3.grid()
    # time_line = ax3.axvline(x=0.0,linewidth=2, color='r')

    ax1.set_xlabel("Behaviour")
    ax1.set_ylabel("Agent")

    ####CULTURE ANIMATION
    def update(i):

        ###AX1
        M = Data["behaviour_value"][i]
        # print("next frame!",M)
        matrice.set_array(M)

        ###AX2
        ax2.clear()
        colour_adjust = norm_neg_pos(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax2,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

        ###AX3
        ax3.clear()
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Culture")

        for i in lines:
            ax3.axhline(y=i, color="b", linestyle="--", alpha=0.3)

        for i in range(int(Data["N"])):
            ax3.plot(Data["network_time"][:i], data[:i])

        ax3.grid()

        # time_line.set_xdata(Data["network_time"][i])

        return matrice

    cbar_behave = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax1,
    )  # This does a mapabble on the fly i think, not sure
    cbar_behave.set_label("Behavioural Value")

    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_neg_pos), ax=ax2
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Culture")

    matrice = ax1.matshow(
        Data["behaviour_value"][0], cmap=cmap_behaviour, aspect="auto"
    )

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(Data["network_time"]),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation_alt.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


def multi_animation_scaled(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], cmap_culture: Union[LinearSegmentedColormap,str], layout: str, node_size:int,  interval:int,
    fps:int, scale_factor: int, frames_proportion: int, norm_neg_pos: SymLogNorm
):

    ####ACUTAL
    frames_list = frame_distribution(
        Data["network_time"], scale_factor, frames_proportion
    )
    # print(frames_list)

    fig = plt.figure(figsize=[7, 7])  # figsize = [8,5]
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Culture")
    data = np.asarray(Data["individual_culture"])  # bodge
    for i in range(int(Data["N"])):
        ax3.plot(Data["network_time"], data[i])

    lines = [-1, -4 / 6, -2 / 6, 0, 2 / 6, 4 / 6, 1]

    for i in lines:
        ax3.axhline(y=i, color="b", linestyle="--", alpha=0.3)

    ax3.grid()
    time_line = ax3.axvline(x=0.0, linewidth=2, color="r")

    ax1.set_xlabel("Behaviour")
    ax1.set_ylabel("Agent")

    ####CULTURE ANIMATION
    def update(i):
        ax2.clear()

        colour_adjust = norm_neg_pos(Data["individual_culture"][frames_list[i]])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax2,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

        M = Data["behaviour_value"][frames_list[i]]
        # print("next frame!",M)
        matrice.set_array(M)

        time_line.set_xdata(Data["network_time"][frames_list[i]])

        return matrice, time_line

    cbar_behave = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax1,
    )  # This does a mapabble on the fly i think, not sure
    cbar_behave.set_label("Behavioural Value")

    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_neg_pos), ax=ax2
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Culture")

    matrice = ax1.matshow(
        Data["behaviour_value"][0], cmap=cmap_behaviour, aspect="auto"
    )

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames_list), repeat_delay=500, interval=interval
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation_scaled.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)


def multiplot_print_average_culture_timeseries(
    FILENAME: str, Data_list:list, seed_list:list, nrows:int, ncols:int, dpi_save:int
):
    y_title = "Average Culture"
    property = "seed_list_print_network_average_culture"
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        ax.plot(Data_list[i][0], Data_list[i][1])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.fill_between(
            Data_list[i][0], Data_list[i][2], Data_list[i][3], alpha=0.3, linewidth=0
        )
        ax.set_title("Seed = {}".format(seed_list[i]))

    plt.tight_layout()

    # colour bar axes
    # fig.subplots_adjust(right=0.8)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def multiplot_print_total_carbon_emissions_timeseries(
    FILENAME: str, Data_list:list, seed_list:list, nrows:int, ncols:int, dpi_save:int
):
    y_title = "Total Carbon Emissions"
    property = "seed_list_print_network_total_carbon_emissions"
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        ax.plot(Data_list[i][0], Data_list[i][1])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Seed = {}".format(seed_list[i]))

    plt.tight_layout()

    # colour bar axes
    # fig.subplots_adjust(right=0.8)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def multiplot_total_carbon_emissions_timeseries(
    FILENAME: str, Data_list:list, seed_list:list, dpi_save:int
):
    y_title = "Total Carbon Emissions"
    property = "seed_list_network_total_carbon_emissions"

    fig, ax = plt.subplots()

    for i in range(len(Data_list)):
        ax.plot(Data_list[i][0], Data_list[i][1])

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)
    ax.set_title("Seed = {}".format(seed_list[i]))

    plt.tight_layout()

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def multiplot_average_culture_timeseries(FILENAME: str, Data_list:list, dpi_save:int):
    y_title = "Average Culture"
    property = "seed_list_network_average_culture"
    fig, ax = plt.subplots()

    for i in range(len(Data_list)):
        ax.plot(Data_list[i][0], Data_list[i][1])
        ax.fill_between(
            Data_list[i][0], Data_list[i][2], Data_list[i][3], alpha=0.3, linewidth=0
        )

    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plt.tight_layout()

    # colour bar axes
    # fig.subplots_adjust(right=0.8)

    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)


def plot_beta_distributions(
    FILENAME:str,
    alpha_attract:float,
    beta_attract:float,
    alpha_threshold:float,
    beta_threshold:float,
    bin_num:int,
    num_counts:int,
    dpi_save:int,
):
    property = "beta_distribution"
    fig, ax = plt.subplots()

    ax.hist(
        np.random.beta(alpha_attract, beta_attract, num_counts),
        bin_num,
        density=True,
        facecolor="g",
        alpha=0.5,
        histtype="stepfilled",
        label="Attract: alpha = "
        + str(alpha_attract)
        + ", beta = "
        + str(beta_attract),
    )
    ax.hist(
        np.random.beta(alpha_threshold, beta_threshold, num_counts),
        bin_num,
        density=True,
        facecolor="b",
        alpha=0.5,
        histtype="stepfilled",
        label="Threshold: alpha = "
        + str(alpha_threshold)
        + ", beta = "
        + str(beta_threshold),
    )
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"PDF")
    ax.legend()
    plotName = FILENAME + "/Plots"
    f = plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi=dpi_save)

def multi_animation_four(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], cmap_culture: Union[LinearSegmentedColormap,str], layout: str, node_size:int,  interval:int,
    fps:int, norm_neg_pos: SymLogNorm,
):

    ####ACUTAL

    fig = plt.figure(figsize=[7, 7])  # figsize = [8,5]
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax3.set_xlabel(r"Time")
    ax3.set_ylabel(r"Culture")

    ax4.plot(Data["network_time"], np.asarray(Data["network_weighting_matrix_convergence"])[0])
    ax4.set_xlabel(r"Time")
    ax4.set_ylabel(r"Change in Agent Link Strength")

    data = np.asarray(Data["individual_culture"])  # bodge
    for i in range(int(Data["N"])):
        ax3.plot(Data["network_time"], data[i])

    #ax3.grid()
    time_line_3 = ax3.axvline(x=0.0, linewidth=2, color="r")
    time_line_4 = ax4.axvline(x=0.0, linewidth=2, color="r")

    ax1.set_xlabel("Behaviour")
    ax1.set_ylabel("Agent")

    ####CULTURE ANIMATION
    def update(i):
        ax2.clear()

        colour_adjust = norm_neg_pos(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax2,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

        M = Data["behaviour_value"][i]
        # print("next frame!",M)
        matrice.set_array(M)

        time_line_3.set_xdata(Data["network_time"][i])
        time_line_4.set_xdata(Data["network_time"][i])

        return matrice, time_line_3,time_line_4

    cbar_behave = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),
        ax=ax1,
    )  # This does a mapabble on the fly i think, not sure
    cbar_behave.set_label("Behavioural Value")

    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_neg_pos), ax=ax2
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Culture")

    matrice = ax1.matshow(
        Data["behaviour_value"][0], cmap=cmap_behaviour, aspect="auto"
    )

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(Data["network_time"]),
        repeat_delay=500,
        interval=interval,
    )

    plt.tight_layout()
    
    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation_four.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

def plot_carbon_emissions_total_orderliness(fileName: str, Data_list: list[Network], dpi_save:int ):
    y_title = "Total Emissions"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_total_carbon_emissions), label = Data_list[i].orderliness)
        ax.set_xlabel(r"Time")
    #ax.axvline(culture_momentum, color='r',linestyle = "--")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_total_emissions_orderliness.png"
    fig.savefig(f, dpi=dpi_save)

def plot_weighting_convergence_orderliness(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "Weighting matrix convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_weighting_matrix_convergence), label = Data_list[i].orderliness)
        ax.set_xlabel(r"Time")
    #ax.axvline(culture_momentum, color='r',linestyle = "--")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_weighting_matrix_convergence_orderliness.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_time_series_orderliness(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = "Culture"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray(Data_list[i].history_time), np.asarray(v.history_culture))

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Orderliness = {}".format(Data_list[i].orderliness))
        #ax.axvline(culture_momentum, color='r',linestyle = "--")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/print_culture_time_series_orderliness.png"
    fig.savefig(f, dpi=dpi_save)



def print_intial_culture_networks_orderliness(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int , layout: str, norm_neg_pos, cmap, node_size):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        pos_culture_network = prod_pos(layout, G)
        # print(i,ax)
        ax.set_title("Orderliness = {}".format(Data_list[i].orderliness))

        indiv_culutre_list = [v.history_culture[0] for v in Data_list[i].agent_list]
        #print(indiv_culutre_list)
        colour_adjust = norm_neg_pos(indiv_culutre_list)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

    plotName = fileName + "/Plots"
    f = plotName + "/print_intial_culture_networks_orderliness.png"
    fig.savefig(f, dpi=dpi_save)

def prints_init_weighting_matrix_orderliness(
    fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int, cmap, 
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data["network_weighting_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        ax.matshow(
            Data_list[i].history_weighting_matrix[0],
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title("orderliness = {}".format(Data_list[i].orderliness))
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")
    plt.tight_layout()

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),
        cax=cbar_ax,
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Weighting matrix")

    plotName = fileName + "/Plots"
    f = plotName + "/prints_init_weighting_matrix_orderliness.png"
    fig.savefig(f, dpi=dpi_save)


def print_intial_culture_networks_homophily_fischer(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int , layout: str, norm_neg_pos, cmap, node_size):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        pos_culture_network = prod_pos(layout, G)
        # print(i,ax)
        ax.set_title("inverse_homophily = {}".format(Data_list[i].inverse_homophily))

        indiv_culutre_list = [v.history_culture[0] for v in Data_list[i].agent_list]
        #print(indiv_culutre_list)
        colour_adjust = norm_neg_pos(indiv_culutre_list)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

    plotName = fileName + "/Plots"
    f = plotName + "/print_intial_culture_networks_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_time_series_homophily_fischer(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = "Culture"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray(Data_list[i].history_time), np.asarray(v.history_culture))

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("inverse homophily = {}".format(Data_list[i].inverse_homophily))
        #ax.axvline(culture_momentum, color='r',linestyle = "--")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/print_culture_time_series_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def prints_culture_network_homophily_fischer(
    FILENAME: str, Data: DataFrame,layout:str, cmap_culture: LinearSegmentedColormap,node_size:int, nrows:int, ncols:int, norm_neg_pos: SymLogNorm, frames_list:list[int], round_dec:int, dpi_save:int
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    # need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data.history_weighting_matrix[0])

    # get pos
    pos_culture_network = prod_pos(layout, G)

    for i, ax in enumerate(axes.flat):
        # print(i,ax)
        ax.set_title(
            "Time= {}".format(round(Data.history_time[frames_list[i]], round_dec))
        )

        culture_list = [x.history_culture[frames_list[i]] for x in Data.agent_list]
        print(culture_list)
        colour_adjust = norm_neg_pos(np.asarray(culture_list))
        #colour_adjust = (Data["individual_culture"][frames_list[i]] + 1)/2
        #colour_adjust = Data["individual_culture"][frames_list[i]]
        ani_step_colours = cmap_culture(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )

    plt.tight_layout()

    # print("cmap_culture", cmap_culture)

    # colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-1, vmax=1)), cax=cbar_ax
    )  # This does a mapabble on the fly i think, not sure
    cbar.set_label("Culture")

    #f = FILENAME + "/Plots/prints_culture_network_homophily_fischer.png"
    #fig.savefig(f, dpi=dpi_save)
