from run import run
from plot import multiplot_print_average_culture_timeseries,multiplot_print_total_carbon_emissions_timeseries,multiplot_total_carbon_emissions_timeseries,multiplot_average_culture_timeseries
from utility import loadData,produceName_random,createFolder
import matplotlib.pyplot as plt
import time
import numpy as np

Y = 3#number of behaviours
nrows_behave = 1
ncols_behave = 3

P = 50#number of agents
K = 5 #k nearest neighbours
prob_wire = 0.1 #re-wiring probability?
behaviour_cap = 1
total_time = 10
delta_t = 0.01#time step size
time_steps_max = int(total_time/delta_t)#number of time steps max, will stop if culture converges
culture_var_min = 0.01#amount of cultural variation
set_seed = 2##reproducibility

#calc culture parameters
culture_momentum = 1#number of time steps used in calculating culture
culture_div = 0#where do we draw the lien for green culture

#Infromation provision parameters
nu = 1# how rapidly extra gains in attractiveness are made
eta = 1#decay rate of information provision boost
attract_information_provision_list = [1]*Y#
t_IP_matrix = [[],[],[],[],[],[]] #list of lists stating at which time steps an information provision policy should be aplied for each behaviour

#Individual learning rate
psi = 1#

#Carbon price parameters
carbon_price_init = 0#
social_cost_carbon = 0
carbon_price_gradient = social_cost_carbon/time_steps_max# social cost of carbon/total time
carbon_emissions = [0.3,0.6,0.8]#np.random.random_sample(Y)#[1]*Y# these should based on some paramters

###PLOT STUFF

round_dec = 2
loadBooleanCSV = ["network_total_carbon_emissions","network_time","network_average_culture","network_min_culture","network_max_culture"]#"network_cultural_var",
loadBooleanArray = []
paramList = []

set_seed_list = [1,2,3,4,5,6,7,8,9]#
nrows = 3
ncols = 3

if __name__ == "__main__":
    RUN = True
    PLOT = True
    SHOW_PLOT = True

    #MULTIRUN_FILENAME = produceName_random(P, K, prob_wire,behaviour_cap,delta_t,Y,culture_var_min,culture_div,nu, eta)
    
    if RUN == False:
        file_name_list = ["results/multirun_random_network_50_5_0.1_1001_1_0.01_1_3_0.01_0_1_1","results/multirun_random_network_50_5_0.1_1001_1_0.01_2_3_0.01_0_1_1"]
        MULTIRUN_FILENAME = "results/multirun_random_network_50_5_0.1_1_0.01_3_0.01_0_1_1"
        createFolder(MULTIRUN_FILENAME)#just_in_case
        file_name_list = ["results/network_50_5_0.1_1001_1_0.01_1_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_2_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_3_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_4_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_5_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_6_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_7_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_8_3_0.01_0_1_1","results/network_50_5_0.1_1001_1_0.01_9_3_0.01_0_1_1"]
        
    else:
        ###RUN MULTIPLE RUNS
        file_name_list = []

        MULTIRUN_FILENAME = produceName_random(P, K, prob_wire,behaviour_cap,delta_t,Y,culture_var_min,culture_div,nu, eta)
        print("MULTIRUN_FILENAME",MULTIRUN_FILENAME)
        
        createFolder(MULTIRUN_FILENAME)
        
        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        for i in range(len(set_seed_list)):
            set_seed = set_seed_list[i]
            FILENAME = run(time_steps_max, culture_var_min, P, K, prob_wire, delta_t, Y, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions)
            file_name_list.append(FILENAME)

        #print("FILE_NAME_LIST: [" + list(file_name_list +"]")
        print ("RUN time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
    
    if PLOT:
        ###GET DATA

        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        data_list_culture = []
        data_list_carbon = []

        for i in file_name_list:
            dataName = i + "/Data"
            Data = loadData(dataName, loadBooleanCSV,loadBooleanArray)
            #Data = get_run_properties(Data,i,paramList)

            Data["network_time"] = np.asarray(Data["network_time"])[0]#for some reason pandas does weird shit
            Data["network_average_culture"] = np.asarray(Data["network_average_culture"])[0]#for some reason pandas does weird shit
            Data["network_min_culture"] = np.asarray(Data["network_min_culture"])[0]#for some reason pandas does weird shit
            Data["network_max_culture"] = np.asarray(Data["network_max_culture"])[0]#for some reason pandas does weird shit
            Data["network_total_carbon_emissions"] = np.asarray(Data["network_total_carbon_emissions"])[0]#for some reason pandas does weird shit

            data_list_culture.append([Data["network_time"],Data["network_average_culture"],Data["network_min_culture"],Data["network_max_culture"]])
            data_list_carbon.append([Data["network_time"],Data["network_total_carbon_emissions"]])

        ###PLOT
        #print("HERE")
        multiplot_print_average_culture_timeseries(MULTIRUN_FILENAME,data_list_culture,set_seed_list,nrows,ncols)#FILENAME,Data_list,seed_list,nrows,ncols
        multiplot_print_total_carbon_emissions_timeseries(MULTIRUN_FILENAME,data_list_carbon,set_seed_list,nrows,ncols)
        multiplot_total_carbon_emissions_timeseries(MULTIRUN_FILENAME,data_list_carbon,set_seed_list)
        multiplot_average_culture_timeseries(MULTIRUN_FILENAME,data_list_culture,set_seed_list)

        print ("PLOT time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

    if SHOW_PLOT:
        plt.show()




