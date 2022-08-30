from logging import raiseExceptions
from run import run
from plot import (
    plot_culture_timeseries,
    animate_weighting_matrix,
    animate_behavioural_matrix,
    animate_culture_network,
    prints_behavioural_matrix,
    prints_culture_network,
    multi_animation,
    multi_animation_alt,
    multi_animation_scaled,
    plot_value_timeseries,
    plot_threshold_timeseries,
    plot_attract_timeseries,
    standard_behaviour_timeseries_plot,
    plot_carbon_price_timeseries,
    plot_total_carbon_emissions_timeseries,
    plot_av_carbon_emissions_timeseries,
    prints_weighting_matrix,
    plot_weighting_matrix_convergence_timeseries,
    plot_cultural_range_timeseries,
    plot_average_culture_timeseries,
    plot_beta_distributions,
    print_network_social_component_matrix,
    animate_network_social_component_matrix,
    animate_network_information_provision,
    print_network_information_provision,
    multi_animation_four,
    print_culture_histogram,
    animate_culture_network_and_weighting,
    weighting_link_timeseries_plot,
    Euclidean_cluster_plot,
    plot_k_cluster_scores,
    plot_behaviour_scatter,
    animate_behaviour_scatter,
)
from utility import loadData, get_run_properties, frame_distribution_prints,k_means_calc,loadObjects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, Normalize
from matplotlib.cm import get_cmap
import time
import numpy as np


# Params
save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False#if true use the exponential form instead like theo
homophily_state = True
alpha_change = True
averaging_method = "Arithmetic"#"Threshold weighted arithmetic"

compression_factor = 10

#Social emissions model
K = 10  # k nearest neighbours INTEGER
M = 3 # number of behaviours
N = 100 # number of agents

total_time = 500

delta_t = 0.05  # time step size
culture_momentum_real = 10# real time over which culture is calculated for INTEGER, NEEDS TO BE MROE THAN DELTA t

prob_rewire = 0.1  # re-wiring probability?

alpha_attract = 1#2  ##inital distribution parameters - doing the inverse inverts it!
beta_attract = 1#3
alpha_threshold = 1#3
beta_threshold = 1#2

time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges
#print("time steps max" , time_steps_max)
set_seed = 1  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 0.1,1
learning_error_scale = 0.02  # 1 standard distribution is 2% error
carbon_emissions = [1]*M

inverse_homophily = 0.2#0.2
homophilly_rate = 1

discount_factor = 0.6
present_discount_factor = 0.8

confirmation_bias = 50

#Infromation provision parameters
if information_provision_state:
    nu = 1# how rapidly extra gains in attractiveness are made
    eta = 0.2#decay rate of information provision boost
    attract_information_provision_list = np.array([0.5*(1/delta_t)]*M)#
    t_IP_matrix = np.array([[],[],[]]) #REAL TIME; list stating at which time steps an information provision policy should be aplied for each behaviour

#Carbon price parameters
if carbon_price_state:
    carbon_price_policy_start = 5#in simualation time to start the policy
    carbon_price_init = 0.0#
    #social_cost_carbon = 0.5
    carbon_price_gradient = 0#social_cost_carbon/time_steps_max# social cost of carbon/total time

params = {
    "opinion_dynamics": opinion_dynamics,
    "save_data": save_data, 
    "compression_factor": compression_factor,
    "time_steps_max": time_steps_max, 
    "carbon_price_state" : carbon_price_state,
    "information_provision_state" : information_provision_state,
    "linear_alpha_diff_state": linear_alpha_diff_state,
    "homophily_state": homophily_state,
    "alpha_change" : alpha_change,
    "averaging_method": averaging_method,
    "delta_t": delta_t,
    "phi_list_lower": phi_list_lower,
    "phi_list_upper": phi_list_upper,
    "N": N,
    "M": M,
    "K": K,
    "prob_rewire": prob_rewire,
    "set_seed": set_seed,
    "culture_momentum_real": culture_momentum_real,
    "learning_error_scale": learning_error_scale,
    "alpha_attract": alpha_attract,
    "beta_attract": beta_attract,
    "alpha_threshold": alpha_threshold,
    "beta_threshold": beta_threshold,
    "carbon_emissions" : carbon_emissions,
    "discount_factor": discount_factor,
    "inverse_homophily": inverse_homophily,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : homophilly_rate,
    "present_discount_factor": present_discount_factor,
    "confirmation_bias": confirmation_bias,
}

if carbon_price_state:
    params["carbon_price_init"] = carbon_price_init
    params["carbon_price_policy_start"] =  carbon_price_policy_start
    params["carbon_price_gradient"] =  carbon_price_gradient

if information_provision_state:
    params["nu"] = nu
    params["eta"] =  eta
    params["attract_information_provision_list"] = attract_information_provision_list
    params["t_IP_matrix"] =  t_IP_matrix


params_name = [#THOSE USEd to create the save list?
    opinion_dynamics,
    time_steps_max,
    M,
    N,
    delta_t,
    K,
    prob_rewire,
    set_seed,
    learning_error_scale,
    alpha_attract,
    beta_attract,
    alpha_threshold,
    beta_threshold,
    culture_momentum_real
]

# SAVING DATA
# THINGS TO SAVE

data_save_behaviour_array_list = ["value", "attract", "threshold"]
data_save_individual_list = ["culture", "carbon_emissions"]
data_save_network_list = [
    "time",
    "cultural_var",
    "total_carbon_emissions",
    "weighting_matrix_convergence",
    "average_culture",
    "min_culture",
    "max_culture",
] 
data_save_network_array_list = [
    "weighting_matrix",
    "social_component_matrix",
]

if information_provision_state:
    data_save_behaviour_array_list.append( "information_provision")

if carbon_price_state:
    data_save_network_list.append("carbon_price")

to_save_list = [
    data_save_behaviour_array_list,
    data_save_individual_list,
    data_save_network_list,
    data_save_network_array_list,
]

# LOAD DATA
paramList = [
    "opinion_dynamics",
    "time_steps_max",
    "M",
    "N",
    "delta_t",
    "K",
    "prob_rewire",
    "set_seed",
    "learning_error_scale",
    "alpha_attract",
    "beta_attract",
    "alpha_threshold",
    "beta_threshold",
    "culture_momentum_real",
]

loadBooleanCSV = [
    "individual_culture",
    "individual_carbon_emissions",
    "network_total_carbon_emissions",
    "network_time",
    "network_cultural_var",
    "network_weighting_matrix_convergence",
    "network_average_culture",
    "network_min_culture",
    "network_max_culture",
]  # "network_cultural_var",,"network_carbon_price"
loadBooleanArray = [
    "network_weighting_matrix",
    "network_social_component_matrix",
    "behaviour_value",
    "behaviour_threshold",
    "behaviour_attract",

]
if information_provision_state:
    loadBooleanArray.append("behaviour_information_provision")
    
if carbon_price_state:
    loadBooleanCSV.append("network_carbon_price")

###PLOT STUFF
nrows_behave = 1
ncols_behave = M
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "white", "olivedrab"])
#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)
#log_norm = SymLogNorm(linthresh=0.15, linscale=1, vmin=-1.0, vmax=1.0, base=10)  # this works at least its correct
cmap_weighting = "Reds"
cmap_edge = get_cmap("Greys")
fps = 5
interval = 50
layout = "circular"
round_dec = 2

nrows = 2
ncols = 3

#print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1
scale_factor = time_steps_max*2

min_val = 1e-3

bin_num = 1000
num_counts = 100000
bin_num_agents = int(round(N/10))
dpi_save = 2000

min_k,max_k = 2,N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

RUN = True
LIVE_DATA = True
LOAD_DATA = True
PLOT = True
cluster_plots = False
SHOW_PLOT = True

frames_list_exponetial = False

if __name__ == "__main__":

    if RUN == False:
        FILENAME = "results/_DEGROOT_200_3_50_0.1_10_0.1_1_0.02_1_1_1_1_1"
    else:
        # start_time = time.time()
        # print("start_time =", time.ctime(time.time()))
        ###RUN MODEL
        #print("start_time =", time.ctime(time.time()))
        FILENAME = run(params, to_save_list, params_name)
        # print ("RUN time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

    if PLOT:
        start_time = time.time()
        print("start_time =", time.ctime(time.time()))

        dataName = FILENAME + "/Data"

        if LIVE_DATA:
            "LOAD LIVE OBJECT"
            live_network = loadObjects(dataName)

        if LOAD_DATA:
            "LOAD DATA"
            Data = loadData(dataName, loadBooleanCSV, loadBooleanArray)
            Data = get_run_properties(Data, FILENAME, paramList)


        #####BODGES!!
        Data["network_time"] = np.asarray(Data["network_time"])[
            0
        ]  # for some reason pandas does weird shit

        
        if frames_list_exponetial: 
            frames_proportion = int(round(len(Data["network_time"]) / 2))
            frames_list = frame_distribution_prints( Data["network_time"], scale_factor, frame_num )
        else:
            #print(len(Data["network_weighting_matrix"]))
            #print(Data["network_time"], len(Data["network_time"]))
            frames_list = [int(round(x)) for x in np.linspace(0, len(Data["network_time"])-1 , num=frame_num + 1)]# -1 is so its within range as linspace is inclusive
             
        print("frames prints:",frames_list)

        ###PLOTS
        #plot_beta_distributions(FILENAME,alpha_attract,beta_attract,alpha_threshold,beta_threshold,bin_num,num_counts,dpi_save,)
        plot_culture_timeseries(FILENAME, Data, dpi_save)
        #plot_value_timeseries(FILENAME,Data,nrows_behave, ncols_behave,dpi_save)
        #plot_threshold_timeseries(FILENAME,Data,nrows_behave, ncols_behave,dpi_save)
        plot_attract_timeseries(FILENAME, Data, nrows_behave, ncols_behave, dpi_save)
        #plot_total_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        #plot_av_carbon_emissions_timeseries(FILENAME, Data, dpi_save)
        #plot_weighting_matrix_convergence_timeseries(FILENAME, Data, dpi_save)
        #plot_cultural_range_timeseries(FILENAME, Data, dpi_save)
        #plot_average_culture_timeseries(FILENAME,Data,dpi_save)
        #weighting_link_timeseries_plot(FILENAME, Data, "Link strength", dpi_save,min_val)
        #plot_behaviour_scatter(FILENAME,Data,"behaviour_attract",dpi_save)
        
        if carbon_price_state:
            plot_carbon_price_timeseries(FILENAME,Data,dpi_save)

        if cluster_plots:
            k_clusters,win_score, scores = k_means_calc(Data,min_k,max_k,size_points)#CALCULATE THE OPTIMAL NUMBER OF CLUSTERS USING SILOUTTE SCORE, DOENST WORK FOR 1
            #k_clusters = 2 # UNCOMMENT TO SET K MANUALLY
            Euclidean_cluster_plot(FILENAME, Data, k_clusters,alpha_val,min_culture_distance, dpi_save)

        ###PRINTS
        
        #prints_weighting_matrix(FILENAME,Data,cmap_weighting,nrows,ncols,frames_list,round_dec,dpi_save)
        #prints_behavioural_matrix(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec,dpi_save)
        #prints_culture_network(FILENAME,Data,layout,cmap,node_size,nrows,ncols,norm_neg_pos,frames_list,round_dec,dpi_save,norm_zero_one)
        #print_network_social_component_matrix(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec,dpi_save)
        #print_culture_histogram(FILENAME, Data, "individual_culture", nrows, ncols, frames_list,round_dec,dpi_save, bin_num_agents)
        if information_provision_state:
            print_network_information_provision(FILENAME,Data,cmap,nrows,ncols,frames_list,round_dec,dpi_save)

        ###ANIMATIONS
        #ani_a = animate_network_information_provision(FILENAME,Data,interval,fps,round_dec,cmap_weighting)
        #ani_b = animate_network_social_component_matrix(FILENAME,Data,interval,fps,round_dec,cmap)
        #ani_c = animate_weighting_matrix(FILENAME,Data,interval,fps,round_dec,cmap_weighting)
        #ani_d = animate_behavioural_matrix(FILENAME,Data,interval,fps,cmap,round_dec)
        #ani_e = animate_culture_network(FILENAME,Data,layout,cmap,node_size,interval,fps,norm_neg_pos,round_dec)
        #ani_f =  animate_culture_network_and_weighting(FILENAME,Data,layout,cmap,node_size,interval,fps,norm_neg_pos,round_dec,cmap_edge)
        #ani_h = multi_animation(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,norm_neg_pos,norm_zero_one)
        #ani_i = multi_animation_four(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,norm_neg_pos)
        #ani_j = multi_animation_alt(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,norm_neg_pos)
        #ani_k = multi_animation_scaled(FILENAME,Data,cmap,cmap,layout,node_size,interval,fps,scale_factor,frames_proportion,norm_neg_pos)
        #ani_l = animate_behaviour_scatter(FILENAME,Data,"behaviour_attract",norm_zero_one, cmap,interval,fps,round_dec)
        print(
            "PLOT time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )



    if SHOW_PLOT:
        plt.show()
