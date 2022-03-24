from pickle import FALSE
from run import run
from plot import plot_culture_timeseries, animate_weighting_matrix, animate_behavioural_matrix, animate_culture_network, prints_behavioural_matrix, prints_culture_network
from utility import loadData, get_run_properties
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

if __name__ == "__main__":

    RUN = True
    PLOT = True
    SHOW_PLOT = False

    if RUN == False:
        FILENAME = "Results/network_12_3_0.1_21_1_0.1_1_6"

    if RUN:
        ###RUN
        P = 200#number of agents
        K = 10 #k nearest neighbours
        prob_wire = 0.1 #re-wiring probability?
        behaviour_cap = 1
        delta_t = 0.1#time step size
        time_steps_max = 1000#number of time steps max, will stop if culture converges
        culture_var_min = 0.01#amount of cultural variation
        set_seed = 1#reproducibility

        name_list = ["pro_env_fuel", "anti_env_fuel","pro_env_transport", "anti_env_transport","pro_env_diet", "anti_env_diet"]
        behave_type_list = [1,0,1,0,1,0]
        Y = len(name_list)#number of behaviours

        start_time = time.time()
        print("start_time =", time.ctime(time.time()))
        ###RUN MODEL
        FILENAME = run(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max,culture_var_min)

        print ("RUN time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

    if PLOT:

        node_size = 100
        cmap_culture = LinearSegmentedColormap.from_list("BrownGreen", ["sienna","white","olivedrab"])
        cmap_behaviour = "coolwarm"
        fps = 5
        interval = 300
        layout_type = "spring"

        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        #LOAD DATA
        loadBooleanCSV = ["individual_culture"]#"network_cultural_var",
        loadBooleanArray = ["network_weighting_matrix","network_social_component_matrix","network_behavioural_attract_matrix","behaviour_value", "behaviour_cost", "behaviour_attract"]
        dataName = FILENAME + "/Data"
        #P,  K, prob_wire, steps,behaviour_cap,delta_t,set_seed,Y
        paramList = [ "P",  "K", "prob_wire","steps","behaviour_cap", "delta_t", "set_seed","Y","culture_var_min"]
        Data = loadData(dataName, loadBooleanCSV,loadBooleanArray)
        Data = get_run_properties(Data,FILENAME,paramList)
        
        steps = int(Data["steps"])
        Y = int(Data["Y"])
        K = int(Data["K"])
        P = int(Data["P"])
        time_list = range(steps)
        frames_prints = [0, round(steps*1/5),round(steps*2/5), round(steps*3/5) ,round( steps*4/5), steps-1]
        
        ###PLOT STUFF
        plot_culture_timeseries(FILENAME,Data,time_list,P)#NEED TO FIX!!
        animate_weighting_matrix(FILENAME,Data,steps,interval,fps)
        animate_behavioural_matrix(FILENAME,Data,steps,interval,fps,cmap_behaviour)
        animate_culture_network(FILENAME,Data,layout_type,cmap_culture,node_size,steps,interval,fps)
        prints_behavioural_matrix(FILENAME,Data,frames_prints,cmap_behaviour)
        prints_culture_network(FILENAME,Data,layout_type,cmap_culture,node_size,frames_prints)

        print ("PLOT time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
        
        if SHOW_PLOT:
            plt.show()





        
