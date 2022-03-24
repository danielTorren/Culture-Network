import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import numpy as np


###DEFINE PLOTS
def plot_culture_timeseries(FILENAME,Data,time_list,P):

    ##plot cultural evolution of agents
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Culture')

    ###WORK OUT HOW TO PLOT STUFF
    #print(Data["individual_culture"][0])
    data = np.asarray(Data["individual_culture"])#bodge
    #print(data)
    for i in range(P):
        #print(Data["individual_culture"][i])
        ax.plot(time_list,data[i])

    lines = [-1,-4/6,-2/6,0,2/6,4/6,1 ]

    for i in lines:
        ax.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + "cultural_evolution.png"
    fig.savefig(f, dpi = 600)


#make matrix animation
def animate_weighting_matrix(FILENAME,Data,steps,interval,fps):

    def update(i):
        M = Data["network_weighting_matrix"][i]
        #print("next frame!",M)
        matrice.set_array(M)
        # Set the title
        ax.set_title("Step = {}".format(i))
        return matrice

    fig, ax = plt.subplots()
    matrice = ax.matshow(Data["network_weighting_matrix"][0])
    plt.colorbar(matrice)

    ani = animation.FuncAnimation(fig, update, frames = steps, repeat_delay = 500, interval = interval )

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

#make behaviour evolution plot
def animate_behavioural_matrix(FILENAME,Data,steps,interval,fps,cmap_behaviour):

    def update(i):
        M = Data["behaviour_value"][i]
        #print("next frame!",M)
        matrice.set_array(M)

        # Set the title
        ax.set_title("Step = {}".format(i))

        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel('Behaviour')
    ax.set_ylabel('Agent')

    matrice = ax.matshow(Data["behaviour_value"][0], cmap = cmap_behaviour, aspect='auto')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax )#This does a mapabble on the fly i think, not sure
    cbar.set_label('Behavioural Value')

    ani = animation.FuncAnimation(fig, update, frames = steps, repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "behavioural_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)


def prod_pos(layout_type,network):

    if layout_type == "circular":
        pos_culture_network = nx.circular_layout(network)
    elif layout_type == "spring":
        pos_culture_network = nx.spring_layout(network)
    elif layout_type == "kamada_kawai":
        pos_culture_network = nx.kamada_kawai_layout(network)
    elif layout_type == "planar":
        pos_culture_network = nx.planar_layout(network)
    else:
        raise Exception('Invalid layout given')

    return pos_culture_network

#animation of changing culture
def animate_culture_network(FILENAME,Data,layout,cmap_culture,node_size,steps,interval,fps):

    def update(i, G,pos, ax,cmap_culture):

        ax.clear()
        #print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
        ani_step_colours = cmap_culture(Data["individual_culture"][i])
        nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos, node_size = node_size, edgecolors = "black")
        
        # Set the title
        ax.set_title("Step = {}".format(i))
        
    # Build plot
    fig, ax = plt.subplots()
    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar.set_label('Culture')

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    ani = animation.FuncAnimation(fig, update, frames= steps, fargs=(G, pos_culture_network, ax, cmap_culture), repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "cultural_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

def prints_behavioural_matrix(FILENAME,Data,frames_prints,cmap_behaviour):

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,7))

    for i, ax in enumerate(axes.flat):

        ax.matshow(Data["behaviour_value"][frames_prints[i]], cmap = cmap_behaviour, aspect='auto')
        # Set the title
        ax.set_title("Step = {}".format(frames_prints[i]))
        ax.set_xlabel('Behaviour')
        ax.set_ylabel('Agent')
    plt.tight_layout()

    #colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), cax=cbar_ax )#This does a mapabble on the fly i think, not sure
    cbar.set_label('Behavioural Value')

    plotName = FILENAME + "/Prints"
    f =  plotName + "/" + "prints_behavioural_matrix.png"
    fig.savefig(f, dpi = 600)

def prints_culture_network(FILENAME,Data,layout,cmap_culture,node_size,frames_prints):

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,7))

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    for i, ax in enumerate(axes.flat):
        ax.set_title("Step =  {}".format(frames_prints[i]))
        ani_step_colours = cmap_culture(Data["individual_culture"][frames_prints[i]])
        nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos_culture_network, node_size = node_size, edgecolors = "black")

    plt.tight_layout()

    #colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), cax=cbar_ax)#This does a mapabble on the fly i think, not sure
    cbar.set_label('Culture')
    
    f = FILENAME + "/Prints/prints_culture_network.png"
    fig.savefig(f, dpi = 600)


        
