from run import run
from plot import plot_culture_timeseries, animate_weighting_matrix, animate_behavioural_matrix, animate_culture_network, prints_behavioural_matrix, prints_culture_network,multi_animation,multi_animation_alt, multi_animation_scaled,plot_value_timeseries, plot_threshold_timeseries, plot_attract_timeseries,standard_behaviour_timeseries_plot,plot_carbon_price_timeseries,plot_total_carbon_emissions_timeseries,plot_av_carbon_emissions_timeseries,prints_weighting_matrix,plot_weighting_matrix_convergence_timeseries,plot_cultural_range_timeseries,plot_average_culture_timeseries,plot_beta_distributions
from utility import loadData, get_run_properties,frame_distribution_prints
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,SymLogNorm
import time
import numpy as np

M = 3#number of behaviours
N = 100#number of agents
K = 20 #k nearest neighbours
prob_wire = 0.2 #re-wiring probability?
total_time = 10
delta_t = 0.01#time step size
time_steps_max = int(total_time/delta_t)#number of time steps max, will stop if culture converges
culture_var_min = 0.01#amount of cultural variation
set_seed = 2##reproducibility
np.random.seed(set_seed)

##inital distribution parameters - doing the inverse inverts it!
alpha_attract =  2
beta_attract = 8
alpha_threshold = 8
beta_threshold = 2

#calc culture parameters
culture_momentum = 5#number of time steps used in calculating culture
culture_div = 0#where do we draw the lien for green culture

#Infromation provision parameters
nu = 1# how rapidly extra gains in attractiveness are made
eta = 1#decay rate of information provision boost
attract_information_provision_list = [1]*M#
t_IP_matrix = [[],[],[],[],[],[]] #list of lists stating at which time steps an information provision policy should be aplied for each behaviour

#social learing
phi_list = np.random.random_sample(M)#conscpicous consumption
print("phi_list",phi_list)
learning_error_scale = 0.02#1 standard distribution is 2% error

#Individual learning
psi = 1#Individual learning rate

#Carbon price parameters
carbon_price_init = 0#
social_cost_carbon = 0
carbon_price_gradient = social_cost_carbon/time_steps_max# social cost of carbon/total time


#[0.3,0.6,0.8]#np.random.random_sample(M)#[1]*M# these should based on some paramters

def calc_carbon_emissions(N,M):
    carbon_emissions_const = 1/N
    carbon_emissions = np.random.random_sample(M)
    carbon_emissions_norm = (carbon_emissions/(carbon_emissions.sum()))*carbon_emissions_const
    return carbon_emissions_norm

carbon_emissions = calc_carbon_emissions(N,M)
#print(carbon_emissions)

#LOAD DATA
loadBooleanCSV = ["individual_culture","individual_carbon_emissions","network_total_carbon_emissions","network_time","network_cultural_var","network_carbon_price","network_weighting_matrix_convergence","network_average_culture","network_min_culture","network_max_culture"]#"network_cultural_var",
loadBooleanArray = ["network_weighting_matrix","network_social_component_matrix","network_behavioural_attract_matrix","behaviour_value", "behaviour_threshold", "behaviour_attract"]

              #N,  K, prob_wire, steps,delta_t,set_seed,M,culture_var_min,culture_div,nu, eta,alpha_attract, beta_attract, alpha_threshold, beta_threshold
paramList = [ "N",  "K", "prob_wire","steps", "delta_t", "set_seed","M","culture_var_min","culture_div","nu", "eta","alpha_attract", "beta_attract", "alpha_threshold", "beta_threshold","learning_error_scale"]

###PLOT STUFF
nrows_behave = 1
ncols_behave = 3
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna","white","olivedrab"])
cmap_weighting = "Reds"
fps = 5
interval = 300
layout = "circular"
round_dec = 2
nrows = 3
ncols = 3
frame_num = ncols*nrows - 1
log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)#this works at least its correct
scale_factor = 100
bin_num = 1000
num_counts = 100000

RUN = True
PLOT = True
SHOW_PLOT = True

if __name__ == "__main__":

    if RUN == False:
        FILENAME = "results/network_100_20_0.2_1001_0.01_5_3_0.01_0_1_1_2_8_8_2_0.02"
    else: 
        start_time = time.time()
        print("start_time =", time.ctime(time.time()))
        ###RUN MODEL
        #FILENAME = run(N,  K, prob_wire, delta_t, M, ,set_seed,time_steps_max,culture_var_min,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix,psi,carbon_price_init,carbon_price_gradient,carbon_emissions
        FILENAME = run(time_steps_max, culture_var_min, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale)

        print ("RUN time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

    if PLOT:

        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        dataName = FILENAME + "/Data"
        Data = loadData(dataName, loadBooleanCSV,loadBooleanArray)
        Data = get_run_properties(Data,FILENAME,paramList)

        #####BODGES!!
        Data["network_time"] = np.asarray(Data["network_time"])[0]#for some reason pandas does weird shit

        #frames_prints = [0, round(Data["steps"]*1/5),round(Data["steps"]*2/5), round(Data["steps"]*3/5) ,round(Data["steps"]*4/5), Data["steps"]-1]
        frames_proportion = int(round(Data["steps"]/2))
        frames_list = frame_distribution_prints(Data["network_time"],scale_factor,frame_num)
        
        ###PLOTS
        #plot_beta_distributions(FILENAME,alpha_attract,beta_attract,alpha_threshold,beta_threshold,bin_num,num_counts )
        plot_culture_timeseries(FILENAME,Data)
        plot_value_timeseries(FILENAME,Data,nrows_behave, ncols_behave)
        plot_threshold_timeseries(FILENAME,Data,nrows_behave, ncols_behave)
        plot_attract_timeseries(FILENAME,Data,nrows_behave, ncols_behave)
        plot_carbon_price_timeseries(FILENAME,Data)
        plot_total_carbon_emissions_timeseries(FILENAME,Data)
        plot_av_carbon_emissions_timeseries(FILENAME,Data)
        plot_weighting_matrix_convergence_timeseries(FILENAME,Data)
        plot_cultural_range_timeseries(FILENAME,Data)
        plot_average_culture_timeseries(FILENAME,Data)

        ###PRINTS
        #prints_weighting_matrix(FILENAME,Data,cmap_weighting,nrows,ncols,frames_list,round_dec)
        #prints_behavioural_matrix(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec)
        #prints_culture_network(FILENAME,Data,layout,cmap,node_size,nrows,ncols,log_norm,frames_list,round_dec)

        ###ANIMATIONS
        #animate_weighting_matrix(FILENAME,Data,interval,fps,round_dec,cmap_weighting)
        #animate_behavioural_matrix(FILENAME,Data,interval,fps,cmap,round_dec)
        #animate_culture_network(FILENAME,Data,layout,cmap,node_size,interval,fps,log_norm,round_dec)
        #multi_animation(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,log_norm)
        #multi_animation_alt(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,log_norm)
        #multi_animation_scaled(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,scale_factor,frames_proportion,log_norm)

        print ("PLOT time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
        
        if SHOW_PLOT:
            plt.show()





        
