from network import Network
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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



P = 50#number of agents
K = 4 #k nearest neighbours
prob_wire = 0.1 #re-wiring probability?
time_steps = 50#number of time steps
behaviour_cap = 1
delta_t = 0.1#time step size

RUNNAME = "network_" + str(P) + "_" + str(K) + "_" +  str(prob_wire) + "_" + str(time_steps) + "_" + str(behaviour_cap) +  "_" + str(delta_t)

name_list = ["pro_env_fuel", "anti_env_fuel","pro_env_transport", "anti_env_transport","pro_env_diet", "anti_env_diet"]
behave_type_list = [0,1,0,1,0,1]
Y = len(name_list)#number of behaviours
time_list = range(time_steps + 1)

if __name__ == "__main__":

    gen_data = True
    plots_gen = True
    animate_behavioural_matrix = True
    animate_culture_network = True
    show_plots = True

    animate_weighting_matrix = False

    start_time = time.time()
    print("start_time =", time.ctime(time.time()))

    if gen_data:
        ### CREATE NETWORK
        social_network = Network( P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap)
        ### RUN TIME STEPS
        for time_counter in range(time_steps):
            social_network.advance_time_step()

    ### PLOT STUFF    
    if plots_gen:
        """
        print("weight matrix")
        print(social_network.weighting_matrix)
        print("agent list")
        agent_culture = [x.culture for x in social_network.agent_list]
        print(agent_culture)
        

        #plot the weighting matrix
        subax1 = plt.subplot()
        plt.matshow(social_network.weighting_matrix)

        
        ### plot network
        subax1 = plt.figure()
        nx.draw(social_network.network, with_labels=True, font_weight='bold')
        """

        ##plot cultural evolution of agents
        fig, ax = plt.subplots()
        ax.set_xlabel('Time')
        ax.set_ylabel('Culture')
        for i in range(P):
            ax.plot(time_list,social_network.agent_list[i].history_culture)
        fig.savefig("results/figures/" + RUNNAME + "cultural_evolution.png")

        
        """
        print("first")
        print(social_network.history_weighting_matrix[0])
        print("next")
        print(social_network.history_weighting_matrix[3])
        """

    #make behaviour evolution plot
    if animate_behavioural_matrix:
        
        behavioural_matrix_time = []

        #print("first value of fist behaviour for first time for first agent",social_network.agent_list[0].history_behaviours_list[0][0].value)
        #print("first attract/cost of fist behaviour for first time for first agent",social_network.agent_list[0].history_behaviours_list[0][0].attract,social_network.agent_list[0].history_behaviours_list[0][0].cost)
        
        #print("first value of fist behaviour for 10th time for first agent",social_network.agent_list[0].history_behaviours_list[10][0].value)
        #print("first attract/cost of fist behaviour for 10th time for first agent",social_network.agent_list[0].history_behaviours_list[10][0].attract,social_network.agent_list[0].history_behaviours_list[10][0].cost)

        
        #print([x for x  in time_list])
        for i in time_list: 
            agents_row = []
            for j in range(P):
                #print("printing first behavioru for agents over time:",social_network.agent_list[j].history_behaviours_list[i][0].value)

                row = [x.value for x in social_network.agent_list[j].history_behaviours_list[i]]
                agents_row.append(row)
            behavioural_matrix_time.append(agents_row)

        #print(behavioural_matrix_time)
        behavioural_matrix_time = np.array(behavioural_matrix_time)
        #print(behavioural_matrix_time)

        #print(behavioural_matrix_time[0])
        #print(behavioural_matrix_time[10])

        def update(i):
            M = behavioural_matrix_time[i]
            #print("next frame!",M)
            matrice.set_array(M)

            # Set the title
            ax.set_title("Frame {}".format(i))

            return matrice

        fig, ax = plt.subplots()
        ax.set_xlabel('Behaviour')
        ax.set_ylabel('Agent')
        c_map = plt.get_cmap('coolwarm')
        matrice = ax.matshow(behavioural_matrix_time[0], cmap = c_map, aspect='auto')
        cbar = fig.colorbar(matrice, ax=ax)#This does a mapabble on the fly i think, not sure
        cbar.set_label('Behavioural Value')

        ani = animation.FuncAnimation(fig, update, frames = len(behavioural_matrix_time), repeat_delay = 500, interval = 100)

        #save the video
        f = "results/videos/" + str(RUNNAME) + "behavioural_matrix_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=10) 
        ani.save(f, writer=writervideo)

    #make matrix animation
    if animate_weighting_matrix:

        def update(i):
            M = social_network.history_weighting_matrix[i]
            #print("next frame!",M)
            matrice.set_array(M)
            # Set the title
            ax.set_title("Frame {}".format(i))
            return matrice

        fig, ax = plt.subplots()
        
        matrice = ax.matshow(social_network.history_weighting_matrix[0])
        plt.colorbar(matrice)

        ani = animation.FuncAnimation(fig, update, frames = len(social_network.history_weighting_matrix), repeat_delay = 500, interval = 100)

        #save the video
        f = "results/videos/" + str(RUNNAME) + "_weighting_matrix_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=10) 
        ani.save(f, writer=writervideo)

    #animation of changing culture
    if animate_culture_network:

        cultural_change_list = []
        for i in range(P):
            cultural_change_list.append(social_network.agent_list[i].history_culture)
        cultural_change_transposed = (np.asarray(cultural_change_list)).transpose()
        #print(cultural_change_transposed[0], cultural_change_transposed[5])
        cultural_change_coloupmap_adjusted =  (cultural_change_transposed + behaviour_cap)/2 # needs to be between 0 an 1
        #print(cultural_change_coloupmap_adjusted[0], cultural_change_coloupmap_adjusted[5])

        def update(i, G,pos, ax,cultural_change_transposed,c_map):

            ax.clear()
            ani_step_colours = c_map(cultural_change_transposed[i])
            nx.draw(G, node_color=ani_step_colours, ax=ax, pos=pos)
            
            # Set the title
            ax.set_title("Frame {}".format(i))
            
        # Build plot
        fig, ax = plt.subplots(figsize=(6,4))

        # Create a graph and layout

        pos = nx.spring_layout(social_network.network)
        c_map = plt.get_cmap('BrBG')
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=c_map), ax=ax)#This does a mapabble on the fly i think, not sure
        cbar.set_label('Culture', rotation=270)

        ani = animation.FuncAnimation(fig, update, frames=time_steps, fargs=(social_network.network, pos,ax, cultural_change_transposed, c_map), repeat_delay = 500, interval = 100)

        #save the video
        f = "results/videos/" + str(RUNNAME) + "_cultural_animation.mp4"
        writervideo = animation.FFMpegWriter(fps=10) 
        ani.save(f, writer=writervideo)

    print ("time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
    if show_plots:
        plt.show()




        
