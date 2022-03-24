from utility import loadData, get_run_properties

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import time


###DEFINE PLOTS
def plot_culture_timeseries(FILENAME,Data,time_list):

    ##plot cultural evolution of agents
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Culture')

    ###WORK OUT HOW TO PLOT STUFF
    for i in range(Data["P"]):
        print(Data["individual_culture"][i])
        ax.plot(time_list,Data["individual_culture"][i])

    lines = [-1,-4/6,-2/6,0,2/6,4/6,1 ]

    for i in lines:
        ax.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + "cultural_evolution.png"
    fig.savefig(f, dpi = 600)


#make matrix animation
def animate_weighting_matrix(FILENAME,Data):

    def update(i,data):
        M = data[i]
        #print("next frame!",M)
        matrice.set_array(M)
        # Set the title
        ax.set_title("Step = {}".format(i))
        return matrice

    fig, ax = plt.subplots()
    
    matrice = ax.matshow(Data["network_weighting_matrix"][0])
    plt.colorbar(matrice)

    ani = animation.FuncAnimation(fig, update, frames = Data["steps"], repeat_delay = 500, interval = interval,fargs=(Data["network_weighting_matrix"]))

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

def gen_behavioural_matrix_time(Data):

    print(Data["behaviour_value"])
    behavioural_matrix_time = (np.asarray(Data["behaviour_value"])).transpose()
    print(behavioural_matrix_time)

    return behavioural_matrix_time

#make behaviour evolution plot
def animate_behavioural_matrix(FILENAME,Data):

    def update(i,data):
        M = data[i]
        #print("next frame!",M)
        matrice.set_array(M)

        # Set the title
        ax.set_title("Step = {}".format(i))

        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel('Behaviour')
    ax.set_ylabel('Agent')

    behavioural_matrix_time = gen_behavioural_matrix_time(Data)

    matrice = ax.matshow(behavioural_matrix_time[0], cmap = cmap_behaviour, aspect='auto')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax )#This does a mapabble on the fly i think, not sure
    cbar.set_label('Behavioural Value')

    ani = animation.FuncAnimation(fig, update, frames = Data["steps"], repeat_delay = 500, interval = interval,fargs=(behavioural_matrix_time))

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

def gen_cultural_change(Data):

    print("Data[indivdiual_culture]",Data["indivdiual_culture"])
    cultural_change = (np.asarray(Data["indivdiual_culture"])).transpose()
    print("cultural_change ",cultural_change )

    return cultural_change

#animation of changing culture
def animate_culture_network(FILENAME,Data,layout,cmap_culture,node_size):

    def update(i, G,pos, ax,cultural_change,cmap_culture):

        ax.clear()
        ani_step_colours = cmap_culture(cultural_change[i])
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

    #gen_cultural_change(Data)
    cultural_change = gen_cultural_change(Data)

    ani = animation.FuncAnimation(fig, update, frames= Data["steps"], fargs=(G, pos_culture_network, ax, cultural_change, cmap_culture), repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "_cultural_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

def prints_behavioural_matrix(FILENAME,Data,frames_prints,cmap_behaviour):

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,7))

    behavioural_matrix_time = gen_behavioural_matrix_time(Data)

    for i, ax in enumerate(axes.flat):

        ax.matshow(behavioural_matrix_time[frames_prints[i]], cmap = cmap_behaviour, aspect='auto')
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

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + "prints_behavioural_matrix.png"
    fig.savefig(f, dpi = 600)

def prints_culture_network(FILENAME,Data,layout,cmap_culture,node_size,frames_prints):

    fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,7))

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    #gen_cultural_change(Data)
    cultural_change = gen_cultural_change(Data)

    for i, ax in enumerate(axes.flat):
        ax.set_title("Step =  {}".format(frames_prints[i]))
        ani_step_colours = cmap_culture(cultural_change[frames_prints[i]])
        nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos_culture_network, node_size = node_size, edgecolors = "black")

    plt.tight_layout()

    #colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), cax=cbar_ax)#This does a mapabble on the fly i think, not sure
    cbar.set_label('Culture')
    
    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + "prints_culture_network.png"
    fig.savefig(f, dpi = 600)



########HISTORY

if __name__ == "__main__":

    FILENAME = "results/network_10_3_0.1_20_1_0.1_1_6"

    node_size = 100
    cmap_culture = LinearSegmentedColormap.from_list("BrownGreen", ["sienna","white","olivedrab"])
    cmap_behaviour = "coolwarm"
    fps = 5
    interval = 300
    layout_type = "spring"

    start_time = time.time()
    print("start_time =", time.ctime(time.time()))

    #LOAD DATA
    loadBoolean = ["network_weighting_matrix","network_social_component_matrix","network_cultural_var", "network_behavioural_attract_matrix","individual_culture","behaviour_value", "behaviour_cost", "behaviour_attract"]
    dataName = FILENAME + "/Data"
    paramList = ["steps", "P",  "K", "prob_wire", "delta_t", "Y","behaviour_cap","set_seed"]

    Data = loadData(dataName,loadBoolean)
    Data = get_run_properties(Data,FILENAME,paramList)

    frames_prints = [0, round(Data["steps"]*1/5),round(Data["steps"]*2/5), round(Data["steps"]*3/5) ,round( Data["steps"]*4/5), Data["steps"]-1]
    time_list = range(Data["steps"])

    ###PLOT STUFF

    plot_culture_timeseries(FILENAME,Data,time_list)
    animate_weighting_matrix(FILENAME,Data)
    animate_behavioural_matrix(FILENAME,Data)
    animate_culture_network(FILENAME,Data,layout_type,cmap_culture,node_size)
    prints_behavioural_matrix(FILENAME,Data,frames_prints,cmap_behaviour)
    prints_culture_network(FILENAME,Data,layout_type,cmap_culture,node_size,frames_prints)

    plt.show()

    print ("time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

        
