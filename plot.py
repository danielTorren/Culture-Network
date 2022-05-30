import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize,LogNorm,SymLogNorm
import numpy as np
from utility import frame_distribution,frame_distribution_prints


###DEFINE PLOTS

def prints_behaviour_timeseries_plot(FILENAME,Data,property,y_title,nrows, ncols):
    PropertyData = Data[property].transpose()

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(14,7))

    for i, ax in enumerate(axes.flat):
        for j in range(int(Data["P"])):
            ax.plot(Data["network_time"], PropertyData[i][j])
        ax.set_xlabel(r"Time")
        ax.set_ylabel(r"%s" % y_title)
        ax.set_title(r"Trait %s" % i)
    plt.tight_layout()

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + property + "_prints_timeseries.png"
    fig.savefig(f, dpi = 600)


def standard_behaviour_timeseries_plot(FILENAME,Data,property,y_title):
    PropertyData = Data[property].transpose()

    fig, ax = plt.subplots()
    for i in range(int(Data["P"])):
        for v in range(int(Data["Y"])):
            ax.plot(Data["network_time"], PropertyData[i][v])
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi = 600)

def plot_value_timeseries(FILENAME,Data,nrows, ncols):
    prints_behaviour_timeseries_plot(FILENAME,Data,"behaviour_value","Trait Value",nrows, ncols)

def plot_threshold_timeseries(FILENAME,Data,nrows, ncols):
    prints_behaviour_timeseries_plot(FILENAME,Data,"behaviour_threshold","Threshold",nrows, ncols)

def plot_attract_timeseries(FILENAME,Data,nrows, ncols):
    prints_behaviour_timeseries_plot(FILENAME,Data,"behaviour_attract","Attractiveness",nrows, ncols)

def plot_carbon_price_timeseries(FILENAME,Data):
    y_title = "Carbon Price"
    property = "network_carbon_price"

    fig, ax = plt.subplots()
    #print(Data["network_carbon_price"])
    ax.plot(Data["network_time"], Data["network_carbon_price"])
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"%s" % y_title)

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + property + "_timeseries.png"
    fig.savefig(f, dpi = 600)


def plot_culture_timeseries(FILENAME,Data):

    ##plot cultural evolution of agents
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Culture')

    ###WORK OUT HOW TO PLOT STUFF
    #print(Data["individual_culture"][0])
    data = np.asarray(Data["individual_culture"])#bodge
    #print(data)
    for i in range(int(int(Data["P"]))):
        #print(Data["individual_culture"][i])
        ax.plot(Data["network_time"],data[i])

    #lines = [-1,-4/6,-2/6,0,2/6,4/6,1 ]

    #for i in lines:
    #    ax.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

    plotName = FILENAME + "/Plots"
    f =  plotName + "/" + "cultural_evolution.png"
    fig.savefig(f, dpi = 600)


#make matrix animation
def animate_weighting_matrix(FILENAME,Data,interval,fps):

    def update(i):
        M = Data["network_weighting_matrix"][i]
        #print("next frame!",M)
        matrice.set_array(M)
        # Set the title
        ax.set_title("Time= {}".format(Data["network_time"][i]))
        return matrice

    fig, ax = plt.subplots()
    matrice = ax.matshow(Data["network_weighting_matrix"][0])
    plt.colorbar(matrice)

    ani = animation.FuncAnimation(fig, update, frames = int(Data["steps"]), repeat_delay = 500, interval = interval )

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "weighting_matrix_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

#make behaviour evolution plot
def animate_behavioural_matrix(FILENAME,Data,interval,fps,cmap_behaviour):

    def update(i):
        M = Data["behaviour_value"][i]
        #print("next frame!",M)
        matrice.set_array(M)

        # Set the title
        ax.set_title("Time= {}".format(Data["network_time"][i]))

        return matrice

    fig, ax = plt.subplots()
    ax.set_xlabel('Behaviour')
    ax.set_ylabel('Agent')

    matrice = ax.matshow(Data["behaviour_value"][0], cmap = cmap_behaviour, aspect='auto')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax )#This does a mapabble on the fly i think, not sure
    cbar.set_label('Behavioural Value')

    ani = animation.FuncAnimation(fig, update, frames = int(Data["steps"]), repeat_delay = 500, interval = interval)

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
def animate_culture_network(FILENAME,Data,layout,cmap_culture,node_size,interval,fps,log_norm):

    def update(i, G,pos, ax,cmap_culture):

        ax.clear()
        #print(Data["individual_culture"][i],Data["individual_culture"][i].shape)
        colour_adjust = log_norm(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos, node_size = node_size, edgecolors = "black")
        
        # Set the title
        ax.set_title("Time= {}".format(Data["network_time"][i]))
        
    # Build plot
    fig, ax = plt.subplots()
    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture,norm=log_norm), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar.set_label('Culture')

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    ani = animation.FuncAnimation(fig, update, frames= int(Data["steps"]), fargs=(G, pos_culture_network, ax, cmap_culture), repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "cultural_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

def prints_behavioural_matrix(FILENAME,Data,cmap_behaviour,nrows,ncols,frames_list,round_dec):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(14,7))

    for i, ax in enumerate(axes.flat):

        ax.matshow(Data["behaviour_value"][frames_list[i]], cmap = cmap_behaviour, aspect='auto')
        # Set the title
        ax.set_title("Time= {}".format(round(Data["network_time"][frames_list[i]]),round_dec))
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

def prints_culture_network(FILENAME,Data,layout,cmap_culture,node_size,nrows,ncols,log_norm,frames_list,round_dec):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(14,7))

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    for i, ax in enumerate(axes.flat):
        #print(i,ax)
        ax.set_title("Time= {}".format(round(Data["network_time"][frames_list[i]]),round_dec))
        
        colour_adjust = log_norm(Data["individual_culture"][frames_list[i]])
        #colour_adjust = (Data["individual_culture"][frames_list[i]] + 1)/2
        ani_step_colours = cmap_culture(colour_adjust)

        nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos_culture_network, node_size = node_size, edgecolors = "black")

    plt.tight_layout()

    #print("cmap_culture", cmap_culture)

    #colour bar axes
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture,norm=log_norm), cax=cbar_ax)#This does a mapabble on the fly i think, not sure
    cbar.set_label('Culture')
    
    f = FILENAME + "/Prints/prints_culture_network.png"
    fig.savefig(f, dpi = 600)


def multi_animation(FILENAME,Data,cmap_behaviour,cmap_culture,layout,node_size,interval,fps,log_norm):

    ####ACUTAL


    fig = plt.figure(figsize =[7,7])#figsize = [8,5]
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,1,2)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Culture')
    data = np.asarray(Data["individual_culture"])#bodge
    for i in range(int(Data["P"])):
        ax3.plot(Data["network_time"],data[i])

    lines = [-1,-4/6,-2/6,0,2/6,4/6,1 ]

    for i in lines:
        ax3.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

    ax3.grid()
    time_line = ax3.axvline(x=0.0,linewidth=2, color='r')

    ax1.set_xlabel('Behaviour')
    ax1.set_ylabel('Agent')

        ####CULTURE ANIMATION
    def update(i):
        ax2.clear()

        colour_adjust = log_norm(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(G, node_color=ani_step_colours, ax=ax2, pos=pos_culture_network , node_size = node_size, edgecolors = "black")

        M = Data["behaviour_value"][i]
        #print("next frame!",M)
        matrice.set_array(M)

        time_line.set_xdata(Data["network_time"][i])

        return matrice,time_line


    cbar_behave = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax1 )#This does a mapabble on the fly i think, not sure
    cbar_behave.set_label('Behavioural Value')

    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=log_norm), ax=ax2)#This does a mapabble on the fly i think, not sure
    cbar_culture.set_label('Culture')

    matrice = ax1.matshow(Data["behaviour_value"][0], cmap = cmap_behaviour, aspect='auto')

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    ani = animation.FuncAnimation(fig, update, frames= len(Data["network_time"]), repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

def multi_animation_alt(FILENAME,Data,cmap_behaviour,cmap_culture,layout,node_size,interval,fps,log_norm):

    ####ACUTAL


    fig = plt.figure(figsize =[7,7])#figsize = [8,5]
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,1,2)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Culture')
    data = np.asarray(Data["individual_culture"])#bodge
    lines = [-1,-4/6,-2/6,0,2/6,4/6,1 ]

    for i in lines:
        ax3.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

    ax3.grid()
    #time_line = ax3.axvline(x=0.0,linewidth=2, color='r')

    ax1.set_xlabel('Behaviour')
    ax1.set_ylabel('Agent')

        ####CULTURE ANIMATION
    def update(i):

        ###AX1
        M = Data["behaviour_value"][i]
        #print("next frame!",M)
        matrice.set_array(M)

        ###AX2
        ax2.clear()
        colour_adjust = log_norm(Data["individual_culture"][i])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(G, node_color=ani_step_colours, ax=ax2, pos=pos_culture_network , node_size = node_size, edgecolors = "black")


        ###AX3
        ax3.clear()
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Culture')

        for i in lines:
            ax3.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

        for i in range(int(Data["P"])):
            ax3.plot(Data["network_time"][:i],data[:i])

        ax3.grid()

        #time_line.set_xdata(Data["network_time"][i])

        return matrice


    cbar_behave = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax1 )#This does a mapabble on the fly i think, not sure
    cbar_behave.set_label('Behavioural Value')

    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture,norm=log_norm), ax=ax2)#This does a mapabble on the fly i think, not sure
    cbar_culture.set_label('Culture')

    matrice = ax1.matshow(Data["behaviour_value"][0], cmap = cmap_behaviour, aspect='auto')

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    ani = animation.FuncAnimation(fig, update, frames= len(Data["network_time"]), repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation_alt.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)

def multi_animation_scaled(FILENAME,Data,cmap_behaviour,cmap_culture,layout,node_size,interval,fps,scale_factor,frames_proportion,log_norm):

    ####ACUTAL
    frames_list = frame_distribution(Data["network_time"],scale_factor,frames_proportion)
    #print(frames_list)

    fig = plt.figure(figsize =[7,7])#figsize = [8,5]
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,1,2)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Culture')
    data = np.asarray(Data["individual_culture"])#bodge
    for i in range(int(Data["P"])):
        ax3.plot(Data["network_time"],data[i])

    lines = [-1,-4/6,-2/6,0,2/6,4/6,1 ]

    for i in lines:
        ax3.axhline(y = i, color = 'b', linestyle = '--', alpha=0.3)

    ax3.grid()
    time_line = ax3.axvline(x=0.0,linewidth=2, color='r')

    ax1.set_xlabel('Behaviour')
    ax1.set_ylabel('Agent')

        ####CULTURE ANIMATION
    def update(i):
        ax2.clear()

        colour_adjust = log_norm(Data["individual_culture"][frames_list[i]])
        ani_step_colours = cmap_culture(colour_adjust)
        nx.draw(G, node_color=ani_step_colours, ax=ax2, pos=pos_culture_network , node_size = node_size, edgecolors = "black")

        M = Data["behaviour_value"][frames_list[i]]
        #print("next frame!",M)
        matrice.set_array(M)

        time_line.set_xdata(Data["network_time"][frames_list[i]])

        return matrice,time_line


    cbar_behave = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-Data["behaviour_cap"], vmax=Data["behaviour_cap"])), ax=ax1 )#This does a mapabble on the fly i think, not sure
    cbar_behave.set_label('Behavioural Value')

    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
    cbar_culture = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture,norm=log_norm), ax=ax2)#This does a mapabble on the fly i think, not sure
    cbar_culture.set_label('Culture')

    matrice = ax1.matshow(Data["behaviour_value"][0], cmap = cmap_behaviour, aspect='auto')

    #need to generate the network from the matrix
    G = nx.from_numpy_matrix(Data["network_weighting_matrix"][0])

    #get pos
    pos_culture_network = prod_pos(layout,G)

    ani = animation.FuncAnimation(fig, update, frames= len(frames_list), repeat_delay = 500, interval = interval)

    #save the video
    animateName = FILENAME + "/Animations"
    f = animateName + "/" + "multi_animation_scaled.mp4"
    writervideo = animation.FFMpegWriter(fps=fps) 
    ani.save(f, writer=writervideo)





        
