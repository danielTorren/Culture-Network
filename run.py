from network import Network
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap,LinearSegmentedColormap, Normalize
import numpy as np
import time
### set up stuff
#plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/daniel/ffmpeg/ffmpeg-5.0-essentials_build/bin/'#NEED TO SPECIFCY WHERE ffmpeg is 

#exogenous variables

"""
    cultural_conformist = 1
    information_provistion = 1
    freq_attract = 1
    carbon_price = 1
    time_cost = 1
"""
print("DEV")
P = 50#number of agents
K = 8 #k nearest neighbours
prob_wire = 0.1 #re-wiring probability?
time_steps = 500#number of time steps
behaviour_cap = 1
delta_t = 0.1#time step size

set_seed = 1#reproducibility

RUNNAME = "network_" + str(P) + "_" + str(K) + "_" +  str(prob_wire) + "_" + str(time_steps) + "_" + str(behaviour_cap) +  "_" + str(delta_t) + "_" + str(set_seed)
name_list = ["pro_env_fuel", "anti_env_fuel","pro_env_transport", "anti_env_transport","pro_env_diet", "anti_env_diet"]
behave_type_list = [1,0,1,0,1,0]
Y = len(name_list)#number of behaviours
time_list = range(time_steps + 1)

#plot variables
node_size = 100
cmap_culture = LinearSegmentedColormap.from_list("BrownGreen", ["sienna","white","olivedrab"])
cmap_behaviour = "coolwarm"
fps = 5
interval = 300
layout_type = "spring"
frames_prints = [0, round(time_steps*1/5),round(time_steps*2/5), round(time_steps*3/5) ,round( time_steps*4/5), time_steps-1]

if __name__ == "__main__":

    gen_data = True

    plots_gen = True
    animate_behavioural_matrix = True
    animate_culture_network = True
    prints_behavioural_matrix = True
    prints_culture_network = True

    show_plots = True

    animate_weighting_matrix = False

    start_time = time.time()
    print("start_time =", time.ctime(time.time()))

    if gen_data:
        ### CREATE NETWORK
        social_network = Network( P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed)
        ### RUN TIME STEPS
        for time_counter in range(time_steps):
            social_network.advance_time_step()

        if animate_culture_network or prints_culture_network:
            cultural_change_list = []
            for i in range(P):
                cultural_change_list.append(social_network.agent_list[i].history_culture)
            cultural_change_transposed = (np.asarray(cultural_change_list)).transpose()
            #print(cultural_change_transposed[0], cultural_change_transposed[-1])
            #cultural_change_coloupmap_adjusted =  (cultural_change_transposed + behaviour_cap)/2 # needs to be between 0 an 1
            #print(cultural_change_transposed[-1],cultural_change_transposed[-1] + behaviour_cap,(cultural_change_transposed[-1] + behaviour_cap)/2)
            #print(cultural_change_coloupmap_adjusted[-1])
            #print(cultural_change_coloupmap_adjusted[0], cultural_change_coloupmap_adjusted[5])

            if layout_type == "circular":
                pos_culture_network = nx.circular_layout(social_network.network)
            elif layout_type == "spring":
                pos_culture_network = nx.spring_layout(social_network.network)
            elif layout_type == "kamada_kawai":
                pos_culture_network = nx.kamada_kawai_layout(social_network.network)
            elif layout_type == "planar":
                pos_culture_network = nx.planar_layout(social_network.network)
            else:
                raise Exception('Invalid layout given')

        if animate_behavioural_matrix or prints_behavioural_matrix:
            behavioural_matrix_time = []
            for i in time_list: 
                agents_row = []
                for j in range(P):
                    #print("printing first behavioru for agents over time:",social_network.agent_list[j].history_behaviours_list[i][0].value)

                    row = [x.value for x in social_network.agent_list[j].history_behaviours_list[i]]
                    agents_row.append(row)
                behavioural_matrix_time.append(agents_row)

            #print(behavioural_matrix_time)
            behavioural_matrix_time = np.array(behavioural_matrix_time)

    ### PLOT STUFF    
    if plots_gen:

        ##plot cultural evolution of agents
        fig, ax = plt.subplots()
        ax.set_xlabel('Steps')
        ax.set_ylabel('Culture')
        for i in range(P):
            ax.plot(time_list,social_network.agent_list[i].history_culture)
        fig.savefig("results/figures/" + RUNNAME + "cultural_evolution.png")

    #make matrix animation
    if animate_weighting_matrix:

        def update(i):
            M = social_network.history_weighting_matrix[i]
            #print("next frame!",M)
            matrice.set_array(M)
            # Set the title
            ax.set_title("Step = {}".format(i))
            return matrice

        fig, ax = plt.subplots()
        
        matrice = ax.matshow(social_network.history_weighting_matrix[0])
        plt.colorbar(matrice)

        ani = animation.FuncAnimation(fig, update, frames = len(social_network.history_weighting_matrix), repeat_delay = 500, interval = interval)

        #save the video
        f = "results/videos/" + str(RUNNAME) + "_weighting_matrix_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=fps) 
        ani.save(f, writer=writervideo)

    #make behaviour evolution plot
    if animate_behavioural_matrix:

        def update(i):
            M = behavioural_matrix_time[i]
            #print("next frame!",M)
            matrice.set_array(M)

            # Set the title
            ax.set_title("Step = {}".format(i))

            return matrice

        fig, ax = plt.subplots()
        ax.set_xlabel('Behaviour')
        ax.set_ylabel('Agent')
        #c_map = plt.get_cmap('coolwarm')
        matrice = ax.matshow(behavioural_matrix_time[0], cmap = cmap_behaviour, aspect='auto')
        #cbar = fig.colorbar(matrice, ax=ax,, vmin=behaviour_cap, vmax=behaviour_cap)#This does a mapabble on the fly i think, not sure
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-behaviour_cap, vmax=behaviour_cap)), ax=ax )#This does a mapabble on the fly i think, not sure
        cbar.set_label('Behavioural Value')

        ani = animation.FuncAnimation(fig, update, frames = len(behavioural_matrix_time), repeat_delay = 500, interval = interval)

        #save the video
        f = "results/videos/" + str(RUNNAME) + "behavioural_matrix_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=fps) 
        ani.save(f, writer=writervideo)

    #animation of changing culture
    if animate_culture_network:

        def update(i, G,pos, ax,cultural_change,cmap_culture):

            ax.clear()
            ani_step_colours = cmap_culture(cultural_change[i])
            nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos, node_size = node_size, edgecolors = "black")
            
            # Set the title
            ax.set_title("Step = {}".format(i))
            
        # Build plot
        fig, ax = plt.subplots()
	    #cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture), ax=ax)#This does a mapabble on the fly i think, not sure
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-behaviour_cap, vmax=behaviour_cap)), ax=ax)#This does a mapabble on the fly i think, not sure
        cbar.set_label('Culture')

        ani = animation.FuncAnimation(fig, update, frames=time_steps, fargs=(social_network.network, pos_culture_network, ax, cultural_change_transposed, cmap_culture), repeat_delay = 500, interval = interval)

        #save the video
        f = "results/videos/" + str(RUNNAME) + "_cultural_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=fps) 
        ani.save(f, writer=writervideo)

    if prints_behavioural_matrix:
        fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,7))

        for i, ax in enumerate(axes.flat):
            matrice = ax.matshow(behavioural_matrix_time[frames_prints[i]], cmap = cmap_behaviour, aspect='auto')
            # Set the title
            ax.set_title("Step = {}".format(frames_prints[i]))
            ax.set_xlabel('Behaviour')
            ax.set_ylabel('Agent')
        plt.tight_layout()

        #colour bar axes
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_behaviour, norm=Normalize(vmin=-behaviour_cap, vmax=behaviour_cap)), cax=cbar_ax )#This does a mapabble on the fly i think, not sure
        cbar.set_label('Behavioural Value')
        fig.savefig("results/figures/" + RUNNAME + "prints_behavioural_matrix.png")
        

    if prints_culture_network:
        fig, axes = plt.subplots(nrows=2, ncols=3,figsize=(14,7))

        for i, ax in enumerate(axes.flat):
            ax.set_title("Step =  {}".format(frames_prints[i]))
            ani_step_colours = cmap_culture(cultural_change_transposed[frames_prints[i]])
            nx.draw(social_network.network, node_color=ani_step_colours, ax=ax, pos=pos_culture_network, node_size = node_size, edgecolors = "black")
        plt.tight_layout()

        #colour bar axes
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_culture, norm=Normalize(vmin=-behaviour_cap, vmax=behaviour_cap)), cax=cbar_ax)#This does a mapabble on the fly i think, not sure
        cbar.set_label('Culture')
        fig.savefig("results/figures/" + RUNNAME + "prints_culture_network.png")

    print ("time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
    if show_plots:
        plt.show()




        
