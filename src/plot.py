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

    return ani

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
    return ani

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
    return ani

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
    return ani

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
    return ani

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
            norm=Normalize(vmin=0, vmax=1),
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())  # This does a mapabble on the fly i think, not sure
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1)),ax=axes.ravel().tolist())
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())
    cbar.set_label("Information Provision")

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_behaviour_information_provision_matrix.png"
    fig.savefig(f, dpi=dpi_save)


def prints_culture_network(
    FILENAME: str, Data: DataFrame,layout:str, cmap_culture: LinearSegmentedColormap,node_size:int, nrows:int, ncols:int, norm_neg_pos: SymLogNorm, frames_list:list[int], round_dec:int, dpi_save:int,norm_zero_one
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_zero_one),ax=axes.ravel().tolist())
    cbar.set_label("Culture")

    f = FILENAME + "/Prints/prints_culture_network.png"
    fig.savefig(f, dpi=dpi_save)


def multi_animation(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], cmap_culture: Union[LinearSegmentedColormap,str], layout: str, node_size:int,  interval:int,
    fps:int, norm_neg_pos: SymLogNorm, norm_zero_one,
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
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_zero_one), ax=ax2
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
    return ani

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
    return ani

def multi_animation_scaled(
    FILENAME: str, Data: DataFrame, cmap_behaviour: Union[LinearSegmentedColormap,str], cmap_culture: Union[LinearSegmentedColormap,str], layout: str, node_size:int,  interval:int,
    fps:int, scale_factor: int, frames_proportion: int, norm_zero_one: SymLogNorm
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

        colour_adjust = norm_zero_one(Data["individual_culture"][frames_list[i]])
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
        plt.cm.ScalarMappable(cmap=cmap_culture, norm=norm_zero_one), ax=ax2
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
    return ani



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
    return ani


def print_intial_culture_networks_homophily_fischer(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int , layout: str, norm_zero_one, cmap, node_size,round_dec):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        pos_culture_network = prod_pos(layout, G)
        # print(i,ax)
        ax.set_title("inverse_homophily = {}".format(round(Data_list[i].inverse_homophily, round_dec)))

        indiv_culutre_list = [v.history_culture[0] for v in Data_list[i].agent_list]
        #print(indiv_culutre_list)
        colour_adjust = norm_zero_one(indiv_culutre_list)
        ani_step_colours = cmap(colour_adjust)

        nx.draw(
            G,
            node_color=ani_step_colours,
            ax=ax,
            pos=pos_culture_network,
            node_size=node_size,
            edgecolors="black",
        )
    
    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_zero_one),ax=axes.ravel().tolist())
    cbar.set_label("Culture")

    plotName = fileName + "/Prints"
    f = plotName + "/print_intial_culture_networks_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_time_series_homophily_fischer(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int, round_dec):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = "Culture"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray(Data_list[i].history_time), np.asarray(v.history_culture))

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Inverse homophily = {}".format(round(Data_list[i].inverse_homophily, round_dec)))
        #ax.axvline(culture_momentum, color='r',linestyle = "--")

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_time_series_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def live_link_change_homophily_fischer(fileName: str, Data_list: list[Network], dpi_save:int, round_dec):
    
    fig, ax = plt.subplots()
    y_title = "Total link strength change"

    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_weighting_matrix_convergence), label = "Inverse homophily = {}".format(round(Data_list[i].inverse_homophily, round_dec)))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/live_link_change_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def live_cum_link_change_homophily_fischer(fileName: str, Data_list: list[Network], dpi_save:int, round_dec):
    
    fig, ax = plt.subplots()
    y_title = "Cumulative total link strength change"

    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(np.asarray(Data_list[i].history_weighting_matrix_convergence))
        print("norm",np.asarray(Data_list[i].history_weighting_matrix_convergence))
        print("cum:", cumulative_link_change)
        ax.plot(np.asarray(Data_list[i].history_time), cumulative_link_change, label = "Inverse homophily = {}".format(round(Data_list[i].inverse_homophily, round_dec)))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/live_cum_link_change_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def live_link_change_homophily_fischer_per_agent(fileName: str, Data_list: list[Network], dpi_save:int, round_dec):
    
    fig, ax = plt.subplots()
    y_title = "Total link strength change per agent"

    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_weighting_matrix_convergence)/Data_list[i].N, label = "Inverse homophily = {}".format(round(Data_list[i].inverse_homophily, round_dec)))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/live_link_change_homophily_fischer_per_agent.png"
    fig.savefig(f, dpi=dpi_save)

def live_cum_link_change_homophily_fischer_per_agent(fileName: str, Data_list: list[Network], dpi_save:int,round_dec):
    
    fig, ax = plt.subplots()
    y_title = "Cumulative total link strength change per agent"

    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(np.asarray(Data_list[i].history_weighting_matrix_convergence)/Data_list[i].N)
        #print("norm",np.asarray(Data_list[i].history_weighting_matrix_convergence))
        #print("cum:", cumulative_link_change)
        ax.plot(np.asarray(Data_list[i].history_time), cumulative_link_change, label = "Inverse homophily = {}".format(round(Data_list[i].inverse_homophily, round_dec)))
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
    ax.legend()
    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/live_cum_link_change_homophily_fischer_per_agent.png"
    fig.savefig(f, dpi=dpi_save)

def prints_culture_network_homophily_fischer(
    FILENAME: str, Data: DataFrame,layout:str, cmap_culture: LinearSegmentedColormap,node_size:int, nrows:int, ncols:int, norm_zero_one: SymLogNorm, frames_list:list[int], round_dec:int, dpi_save:int
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
        #print(culture_list)
        colour_adjust = norm_zero_one(np.asarray(culture_list))
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

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-1, vmax=1)),ax=axes.ravel().tolist())
    cbar.set_label("Culture")

    f = FILENAME + "/Prints/prints_culture_network_homophily_fischer.png"
    fig.savefig(f, dpi=dpi_save)

def print_intial_culture_networks_confirmation_bias(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int , layout: str, norm_neg_pos, cmap, node_size, round_dec):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        G = nx.from_numpy_matrix(Data_list[i].history_weighting_matrix[0])
        pos_culture_network = prod_pos(layout, G)
        # print(i,ax)
        ax.set_title("Confirmation bias = {}".format(round(Data_list[i].confirmation_bias, round_dec)))

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

    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_time_series_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)

def prints_init_weighting_matrix_confirmation_bias(
    fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int, cmap, 
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data["network_weighting_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        mat = ax.matshow(
            Data_list[i].history_weighting_matrix[0],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        ax.set_title("Confirmation bias = {}".format(Data_list[i].confirmation_bias))
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")
    plt.tight_layout()

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())  # This does a mapabble on the fly i think, not sure
    
    cbar.set_label("Weighting matrix")

    plotName = fileName + "/Prints"
    f = plotName + "/prints_init_weighting_matrix_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)

def prints_final_weighting_matrix_confirmation_bias(
    fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int, cmap, 
):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data["network_weighting_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        # print(i)
        mat = ax.matshow(
            Data_list[i].history_weighting_matrix[-1],
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
        )
        ax.set_title("Confirmation bias = {}".format(Data_list[i].confirmation_bias))
        ax.set_xlabel("Agent Link Strength")
        ax.set_ylabel("Agent Link Strength")

    plt.tight_layout()
    # colour bar axes
    cbar = fig.colorbar(mat,ax=ax)  # This does a mapabble on the fly i think, not sure
    
    cbar.set_label("Weighting matrix")
    

    plotName = fileName + "/Prints"
    f = plotName + "/prints_final_weighting_matrix_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)


    
def plot_carbon_emissions_total_confirmation_bias(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "Total Emissions"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_total_carbon_emissions), label = Data_list[i].confirmation_bias)
        ax.set_xlabel(r"Time")
    #ax.axvline(culture_momentum, color='r',linestyle = "--")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_total_emissions_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)

def plot_weighting_convergence_confirmation_bias(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "Weighting matrix convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_weighting_matrix_convergence), label = Data_list[i].confirmation_bias)
        ax.set_xlabel(r"Time")
    #ax.axvline(culture_momentum, color='r',linestyle = "--")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_weighting_matrix_convergence_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)

def plot_cum_weighting_convergence_confirmation_bias(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "Weighting matrix convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(np.asarray(Data_list[i].history_weighting_matrix_convergence))
        ax.plot(np.asarray(Data_list[i].history_time),cumulative_link_change , label = Data_list[i].confirmation_bias)
        ax.set_xlabel(r"Time")
    #ax.axvline(culture_momentum, color='r',linestyle = "--")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_cum_weighting_matrix_convergence_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_time_series_confirmation_bias(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = "Culture"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray(Data_list[i].history_time), np.asarray(v.history_culture))

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Confirmation bias = {}".format(Data_list[i].confirmation_bias))
        #ax.axvline(culture_momentum, color='r',linestyle = "--")

    plt.tight_layout()

    plotName = fileName + "/Plots"
    f = plotName + "/print_culture_time_series_confirmation_bias.png"
    fig.savefig(f, dpi=dpi_save)


def multi_animation_weighting(FILENAME: str, data_list: list, cmap: Union[LinearSegmentedColormap,str],  interval:int, fps:int, round_dec:int ,nrows: int, ncols:int, time_steps_max:int):
    ####ACUTAL
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    def update(i, data_list, axes):
        matrices_list = []
        for v, ax in enumerate(axes.flat):
            M = data_list[v].history_weighting_matrix[i]
            # print("next frame!",M)        
            matrice.set_array(M)
            # Set the title
            matrices_list.append(matrice)
        #time_ax.text(0, 0, "Time= {}".format(round(data_list[0].history_time[i])), ha="center", va="center")
        #time_ax.annotate("Time= {}".format(round(data_list[0].history_time[i], round_dec)),xy=(0, 0), ha="center")
        #time_ax.set_title("Time= {}".format(round(data_list[0].history_time[i], round_dec)))
        
        #plt.suptitle("Time= {}".format(round(data_list[0].history_time[i], round_dec)),fontsize=20)

        return matrices_list

    for v, ax in enumerate(axes.flat):
        ax.set_xlabel("Agent")
        ax.set_ylabel("Agent")
        matrice = ax.matshow(data_list[v].history_weighting_matrix[0], cmap=cmap)

    # colour bar axes
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1)),ax=axes.ravel().tolist())
    cbar.set_label("Agent Link Strength")

    plt.suptitle("Time= {}".format(round(data_list[0].history_time[0], round_dec)),fontsize=20)

    ani = animation.FuncAnimation(
        fig,
        update,
        fargs=(data_list,axes),
        frames=int(time_steps_max),
        repeat_delay=500,
        interval=interval,
    )
    #print("ani: ",ani)
    
    #plt.tight_layout()

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)
    return ani
"""WEIGHTING"""

def plot_average_culture_no_range_comparison(fileName: str, Data_list: list[Network], dpi_save:int, property_list:list):
    y_title = "Average Culture"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):

        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_average_culture), label = property_list[i])
        
    ax.set_xlabel(r"Time")
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/plot_average_culture_no_range_comparison.png"
    fig.savefig(f, dpi=dpi_save)

def plot_average_culture_comparison(fileName: str, Data_list: list[Network], dpi_save:int, property_list:list):
    y_title = "Average Culture"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        #print(np.asarray(Data_list[i].history_average_culture))
        culture_min = np.asarray(Data_list[i].history_min_culture)  # bodge
        culture_max = np.asarray(Data_list[i].history_max_culture)  # bodge

        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_average_culture), label = property_list[i])
        
        
        ax.fill_between(
            np.asarray(Data_list[i].history_time), culture_min, culture_max, alpha=0.5, linewidth=0
        )
    ax.set_xlabel(r"Time")
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_av_cultures.png"
    fig.savefig(f, dpi=dpi_save)

def plot_carbon_emissions_total_comparison(fileName: str, Data_list: list[Network], dpi_save:int, property_list:list):
    y_title = "Total Emissions"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_total_carbon_emissions), label = property_list[i])
        ax.set_xlabel(r"Time")
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_total_emissions.png"
    fig.savefig(f, dpi=dpi_save)

def plot_weighting_matrix_convergence_comparison(fileName: str, Data_list: list[Network], dpi_save:int, property_list:list):
    y_title = "weighting_matrix_convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):

        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_weighting_matrix_convergence), label = property_list[i])
        ax.set_xlabel(r"Time")
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_weighting_matrix_convergence.png"
    fig.savefig(f, dpi=dpi_save)

def plot_cum_weighting_matrix_convergence_comparison(fileName: str, Data_list: list[Network], dpi_save:int, property_list:list):
    y_title = "weighting_matrix_convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        cumulative_link_change = np.cumsum(np.asarray(Data_list[i].history_weighting_matrix_convergence))
        ax.plot(np.asarray(Data_list[i].history_time), cumulative_link_change, label = property_list[i])
        ax.set_xlabel(r"Time")
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_cum_weighting_matrix_convergence.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_timeseries(fileName: str, Data_list: list[Network] , title_list:str, nrows:int, ncols:int , dpi_save:int):
    y_title = "indivdiual culture"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):
        ax.set_title(title_list[i])
        ax.set_ylabel(r"%s" % y_title)
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray( Data_list[i].history_time ), np.asarray(v.history_culture))
        ax.set_xlabel(r"Time")
    
    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_timeseries.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_timeseries_vary_conformity_bias(fileName: str, Data_list: list[Network] , conformity_title_list:str, alpha_title_list:str, nrows:int, ncols:int , dpi_save:int):
    """
    y_title = "Indivdiual culture"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        ax.set_title(title_list[i])

        ax.set_ylabel(r"%s" % y_title)

        for v in Data_list[i].agent_list:
            ax.plot(np.asarray( Data_list[i].history_time ), np.asarray(v.history_culture))

        ax.set_xlabel(r"Time")
    
    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_timeseries_vary_conformity_bias.png"
    fig.savefig(f, dpi=dpi_save)
    """


    #####
    y_title = "Indivdiual culture"

    fig = plt.figure(constrained_layout=True)
    #fig.suptitle('Culture timeseries with varying conformity bias and alpha dynamics')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=nrows, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Conformity bias = {conformity_title_list[row]}')

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=ncols)
        for col, ax in enumerate(axs):
            for v in Data_list[row][col].agent_list:
                ax.plot(np.asarray(Data_list[row][col].history_time ), np.asarray(v.history_culture))
            ax.set_title(r'%s' % alpha_title_list[col])
            #ax.set_ylabel(r"%s" % y_title)
            #ax.set_xlabel(r"Time")
    fig.supxlabel(r"Time")
    fig.supylabel(r"%s" % y_title)
    
    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_timeseries_vary_conformity_bias.png"
    fig.savefig(f, dpi=dpi_save)









    """SA"""

def prints_SA_matrix(FILENAME, Data,title_list,cmap,nrows, ncols, dpi_save , labels):


    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print(frames_list,Data["network_time"],Data["network_weighting_matrix"][20])
    #print( len(Data["network_weighting_matrix"]),frames_list)
    for i, ax in enumerate(axes.flat):
        matrix = ax.matshow(
            Data[i],
            cmap=cmap,
            aspect="auto",
        )
        ax.set_title(title_list[i])
        # colour bar axes
        cbar = fig.colorbar(matrix,ax=ax)  # This does a mapabble on the fly i think, not sure
        xaxis = np.arange(len(labels))
        ax.set_xticks(xaxis)
        ax.set_yticks(xaxis)
        ax.set_xticklabels(labels, rotation = 45)
        ax.set_yticklabels(labels, rotation = 45)
        #ax.xticks(rotation=45, ha='right')
    #plt.tight_layout()

    plotName = FILENAME + "/Prints"
    f = plotName + "/" + "prints_SA_matrix.png"
    fig.savefig(f, dpi=dpi_save)

"""OTHER"""

def print_culture_histgram(fileName: str, Data_list: list[Network] , title_list:str, nrows:int, ncols:int , dpi_save:int):
    y_title = "indivdiual culture"

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    for i, ax in enumerate(axes.flat):

        ax.set_title(title_list[i])
        ax.set_ylabel(r"%s" % y_title)

        ax.hist(x, density=True, bins=30)  # density=False would make counts
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray( Data_list[i].history_time ), np.asarray(v.history_culture))
        ax.set_xlabel(r"Time")
    
    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_timeseries.png"
    fig.savefig(f, dpi=dpi_save)



def print_culture_histogram(
    FILENAME: str, Data: DataFrame, property:str, nrows:int, ncols:int, frames_list, round_dec, dpi_save,bin_num
):
    y_title = "Probability"
    #print(Data[property], Data[property].shape)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    # print("property = ", property)

    for i, ax in enumerate(axes.flat):
        #print(Data[property][frames_list[i]])
        ax.hist(Data[property][frames_list[i]], density=True, bins = bin_num)  # density=False would make counts
        ax.set_xlabel(r"Culture")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Time= {}".format(round(Data["network_time"][frames_list[i]], round_dec)))  # avoid 0 in the title
    plt.tight_layout()

    plotName = FILENAME + "/Plots"
    f = plotName + "/print_culture_histogram.png"
    fig.savefig(f, dpi=dpi_save)


# animation of changing culture
def animate_culture_network_and_weighting(
    FILENAME: str, Data: DataFrame, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_neg_pos: SymLogNorm, round_dec:int, cmap_edge
):

    def update(i, G, pos, ax, cmap_culture):

        ax.clear()
        # print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
        
        colour_adjust = norm_neg_pos(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)

        G = nx.from_numpy_matrix( Data["network_weighting_matrix"][i])

        weights = [G[u][v]['weight'] for u,v in G.edges()]
        norm = Normalize(vmin=0, vmax=1)
        colour_adjust_edge = norm(weights)
        colors_weights = cmap_edge(colour_adjust_edge)

        nx.draw(
            G,
            node_color=ani_step_colours,
            edge_color=colors_weights,
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
    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Culture")

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_edge), ax=ax, location='left',
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Link Strength")

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

    return ani

# animation of changing culture
def live_compare_animate_culture_network_and_weighting(
    FILENAME: str, Data_list: list, layout:str, cmap_culture: Union[LinearSegmentedColormap,str], node_size:int, interval:int, fps:int, norm_zero_one: SymLogNorm, round_dec:int, cmap_edge, nrows, ncols,property_name, property_list
):

    def update(i, Data_list, axes, cmap_culture, layout,title ):

        for j, ax in enumerate(axes.flat):

            ax.clear()
            # print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
            individual_culture_list = [x.culture for x in Data_list[j].agent_list]
            colour_adjust = norm_zero_one(individual_culture_list)
            ani_step_colours = cmap_culture(colour_adjust)

            G = nx.from_numpy_matrix(Data_list[j].history_weighting_matrix[i])

             # get pos
            pos = prod_pos(layout, G)

            weights = [G[u][v]['weight'] for u,v in G.edges()]
            norm = Normalize(vmin=0, vmax=1)
            colour_adjust_edge = norm(weights)
            colors_weights = cmap_edge(colour_adjust_edge)

            nx.draw(
                G,
                node_color=ani_step_colours,
                edge_color=colors_weights,
                ax=ax,
                pos=pos,
                node_size=node_size,
                edgecolors="black",
            )

            ax.set_title( r"%s = %s" % (property_name, round(property_list[j], round_dec)))

        title.set_text("Time= {}".format(round(Data_list[0].history_time[i], round_dec)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))

    title = plt.suptitle(t='', fontsize = 20)

    cbar_culture = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_culture), ax=axes.ravel().tolist(), location='right'
    )  # This does a mapabble on the fly i think, not sure
    cbar_culture.set_label("Culture")

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_edge), ax=axes.ravel().tolist(), location='left',
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Link Strength")

    # need to generate the network from the matrix
    #G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])



    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, cmap_culture, layout,title),
        repeat_delay=500,
        interval=interval,
    )


    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/live_multi_animate_culture_network_and_weighting_%s.mp4" % property_name
    #print("f", f)
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

# animation of changing culture
def live_compare_animate_weighting_matrix(
    FILENAME: str, Data_list: list,  cmap_weighting: Union[LinearSegmentedColormap,str], interval:int, fps:int, round_dec:int, cmap_edge, nrows, ncols,property_name, property_list
):

    def update(i, Data_list, axes, title ):

        for j, ax in enumerate(axes.flat):

            ax.clear()

            ax.matshow(Data_list[j].history_weighting_matrix[i], cmap=cmap_weighting, norm=Normalize(vmin=0, vmax=1),aspect="auto" )
            
            ax.set_title( r"%s = %s" % (property_name, round(property_list[j], round_dec)))
            ax.set_xlabel("Agent")
            ax.set_ylabel("Agent")

        title.set_text("Time= {}".format(round(Data_list[0].history_time[i], round_dec)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True)

    #plt.tight_layout()

    title = plt.suptitle(t='', fontsize = 20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_weighting), ax=axes.ravel().tolist(), location='right',
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Link Strength")

    # need to generate the network from the matrix
    #G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])
    

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, title),
        repeat_delay=500,
        interval=interval,
    )


    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/live_compare_animate_weighting_matrix_%s.mp4" % property_name
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani

# animation of changing culture
def live_compare_animate_behaviour_matrix(
    FILENAME: str, Data_list: list,  cmap_behaviour: Union[LinearSegmentedColormap,str], interval:int, fps:int, round_dec:int, nrows, ncols,property_name, property_list
):

    def update(i, Data_list, axes, title ):

        for j, ax in enumerate(axes.flat):

            ax.clear()

            for q in Data_list[j].agent_list:
                q.history_behaviour_values

            M = [n.history_behaviour_values[i] for n in Data_list[j].agent_list]

            ax.matshow(M, cmap=cmap_behaviour, norm=Normalize(vmin=-1, vmax=1),aspect="auto",)
            
            ax.set_title( r"%s = %s" % (property_name, round(property_list[j], round_dec)))
            ax.set_xlabel("Agent")
            ax.set_ylabel("Agent")

        title.set_text("Time= {}".format(round(Data_list[0].history_time[i], round_dec)))

        

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7), constrained_layout=True)

    #fig.tight_layout(h_pad=2)
    #plt.tight_layout()

    title = plt.suptitle(t='', fontsize = 20)

    cbar_weight = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap_behaviour), ax=axes.ravel().tolist(), location='right',
    )  # This does a mapabble on the fly i think, not sure
    cbar_weight.set_label("Behavioural Value")

    # need to generate the network from the matrix
    #G = nx.from_numpy_matrix(Data_list[0].history_weighting_matrix[0])

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=int(len(Data_list[0].history_time)),
        fargs=(Data_list, axes, title),
        repeat_delay=500,
        interval=interval,
    )

    # save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/live_compare_animate_behaviour_matrix_%s.mp4" % property_name
    writervideo = animation.FFMpegWriter(fps=fps)
    ani.save(f, writer=writervideo)

    return ani


def weighting_link_timeseries_plot(FILENAME: str, Data: DataFrame, y_title:str, dpi_save:int, min_val):

    fig, ax = plt.subplots()

    for i in range(int(Data["N"])):
        for v in range(int(Data["N"])):
            if Data["network_weighting_matrix"][0][i][v] > 0.0:
                link_data = [Data["network_weighting_matrix"][x][i][v] for x in range(len(Data["network_time"]))]
                if any(j > min_val for j in link_data):
                    ax.plot(Data["network_time"], link_data)

    ax.axvline(Data["culture_momentum_real"], color='r',linestyle = "--")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f = plotName + "/weighting_link_timeseries_plot.png"
    fig.savefig(f, dpi=dpi_save)

def print_culture_time_series_data_compression(fileName: str, Data_list: list[Network], dpi_save:int,nrows: int, ncols:int, round_dec):
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 7))
    y_title = "Culture"

    for i, ax in enumerate(axes.flat):
        for v in Data_list[i].agent_list:
            ax.plot(np.asarray(Data_list[i].history_time), np.asarray(v.history_culture))

        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title("Data compression = {}".format(round(Data_list[i].compression_factor, round_dec)))
        #ax.axvline(culture_momentum, color='r',linestyle = "--")

    plt.tight_layout()

    plotName = fileName + "/Prints"
    f = plotName + "/print_culture_time_series_data_compression.png"
    fig.savefig(f, dpi=dpi_save)



