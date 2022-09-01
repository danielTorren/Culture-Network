from logging import raiseExceptions
from run import parallel_run
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA,produce_param_list
from matplotlib.colors import LinearSegmentedColormap,  Normalize
from matplotlib.cm import get_cmap
from plot import (
    print_culture_time_series_generic,
    plot_average_culture_comparison,
    plot_carbon_emissions_total_comparison,
    plot_weighting_matrix_convergence_comparison,
    plot_average_culture_no_range_comparison,
    plot_live_link_change_comparison,
    plot_live_cum_link_change_comparison,
    plot_live_link_change_per_agent_comparison,
    plot_live_cum_link_change_per_agent_comparison,
    print_culture_time_series_clusters,
    print_live_intial_culture_networks,
    prints_init_weighting_matrix,
    prints_final_weighting_matrix,
    multi_animation_weighting,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    live_compare_plot_animate_behaviour_scatter,
    live_print_heterogenous_culture_momentum,
)

# Params
save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False #if true use the exponential form instead like theo
homophily_state = True
alpha_change = True
heterogenous_cultural_momentum = True
harsh_data = False

#Social emissions model
K = 12 # k nearest neighbours INTEGER
M = 3  # number of behaviours
N = 30  # number of agents

total_time = 100
culture_momentum_real = 10
delta_t = 0.1  # time step size
averaging_method = "Arithmetic" #"Geometric"#"Arithmetic"#"Threshold weighted arithmetic"

compression_factor = 5

prob_rewire = 0.1  # re-wiring probability?

time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges
#print("time steps max" , time_steps_max)
set_seed = 1  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 0.1,1

learning_error_scale = 0.02  # 1 standard distribution is 2% error

inverse_homophily = 0.3#0.2
homophilly_rate = 1

discount_factor = 0.6#0.6
present_discount_factor = 0.8#0.8

confirmation_bias = 10

if heterogenous_cultural_momentum:
    quick_changers_prop = 0.2#proportion of population that are quick changers that have very light memory
    lagards_prop = 0.2#proportion of population that are quick changers that have very long memory
    ratio_quick_changers = 0.25
    ratio_lagards = 4
    
#harsh data parameters
if harsh_data:
    green_extreme_max = 8#beta distribution value max
    green_extreme_min = 2#beta distribution value min
    green_extreme_prop = 2/5# proportion of population that are green
    indifferent_max = 2
    indifferent_min = 2
    indifferent_prop = 1/5#proportion that are indifferent
    brown_extreme_min = 2
    brown_extreme_max = 8
    brown_extreme_prop = 2/5#proportion that are brown
    if green_extreme_prop + indifferent_prop + brown_extreme_prop != 1:
        print(green_extreme_prop + indifferent_prop + brown_extreme_prop)
        raise Exception("Invalid proportions")
else:
    alpha_attract = 1#2  ##inital distribution parameters - doing the inverse inverts it!
    beta_attract = 1#3
    alpha_threshold = 1#3
    beta_threshold = 1#2

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
    "heterogenous_cultural_momentum" : heterogenous_cultural_momentum,
    "harsh_data": harsh_data,
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
    "discount_factor": discount_factor,
    "inverse_homophily": inverse_homophily,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : homophilly_rate,
    "present_discount_factor": present_discount_factor,
    "confirmation_bias": confirmation_bias,
}

if heterogenous_cultural_momentum:
    params["quick_changers_prop"] = quick_changers_prop
    params["lagards_prop"] = lagards_prop
    params["ratio_quick_changers"] = ratio_quick_changers
    params["ratio_lagards"] = ratio_lagards

#behaviours!
if harsh_data:#trying to create a polarised society!
    params["green_extreme_max"]= green_extreme_max
    params["green_extreme_min"]= green_extreme_min
    params["green_extreme_prop"]= green_extreme_prop
    params["indifferent_max"]= indifferent_max
    params["indifferent_min"]= indifferent_min
    params["indifferent_prop"]= indifferent_prop
    params["brown_extreme_min"]= brown_extreme_min 
    params["brown_extreme_max"]= brown_extreme_max
    params["brown_extreme_prop"]= brown_extreme_prop
else:
    params["alpha_attract"] = alpha_attract
    params["beta_attract"] = beta_attract
    params["alpha_threshold"] = alpha_threshold
    params["beta_threshold"] = beta_threshold


if carbon_price_state:
    params["carbon_price_init"] = carbon_price_init
    params["carbon_price_policy_start"] =  carbon_price_policy_start
    params["carbon_price_gradient"] =  carbon_price_gradient

if information_provision_state:
    params["nu"] = nu
    params["eta"] =  eta
    params["attract_information_provision_list"] = attract_information_provision_list
    params["t_IP_matrix"] =  t_IP_matrix

###PLOT STUFF
nrows_behave = 1
ncols_behave = M
node_size = 50
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "whitesmoke", "olivedrab"], gamma=1)

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

alpha_quick, alpha_normal, alpha_lagard = 0.9,0.7,0.9
colour_quick, colour_normal, colour_lagard = "indianred", "grey", "cornflowerblue"

#print("time_steps_max", time_steps_max)
frame_num = ncols * nrows - 1
scale_factor = time_steps_max*2

min_val = 1e-3

bin_num = 1000
num_counts = 100000
bin_num_agents = int(round(N/10))
dpi_save = 2000

min_k,max_k = 2,10#N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

if __name__ == "__main__":

        nrows = 2
        ncols = 2#due to screen ratio want more cols than rows usually
        reps = nrows*ncols
        if ncols*nrows > reps:
            raiseExceptions("Too many rows or columns for number of repetitions")

        property_varied = "culture_momentum"#"confirmation_bias"#"inverse_homophily" #MAKE SURE ITS TYPES CORRECTLY
        property_varied_title = "Cultural momentum"
        param_min = 4.0
        param_max = 25.0#50.0

    
        fileName = "results/%s_variation_%s_%s_%s_%s_%s_%s" % (property_varied,str(params["N"]),str(params["time_steps_max"]),str(params["K"]), str(param_min), str(param_max), str(reps))
        print("fileName: ", fileName)

        property_values_list = np.linspace(param_min,param_max, reps)
        params_list = produce_param_list(params,property_values_list, property_varied)
        data = parallel_run(params_list)#better if a Multiple of 4

        createFolderSA(fileName)

        ###WORKING 

        print_culture_time_series_generic(fileName, data, property_values_list, property_varied_title, dpi_save,nrows, ncols,round_dec)
        #plot_average_culture_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_carbon_emissions_total_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_weighting_matrix_convergence_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_average_culture_no_range_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_link_change_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_cum_link_change_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_link_change_per_agent_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
        #plot_live_cum_link_change_per_agent_comparison(fileName, data, dpi_save,property_values_list, property_varied,round_dec)
    

        #print_culture_time_series_clusters(fileName, data, property_values_list, property_varied_title, min_k,max_k,size_points, alpha_val, min_culture_distance, nrows, ncols, dpi_save, round_dec)
        #print_live_intial_culture_networks(fileName, data, dpi_save, property_values_list, property_varied, nrows, ncols , layout, norm_zero_one, cmap, node_size,round_dec)
        #prints_init_weighting_matrix(fileName, data, dpi_save,nrows, ncols, cmap_weighting,property_values_list, property_varied,round_dec)
        #prints_final_weighting_matrix(fileName, data, dpi_save,nrows, ncols, cmap_weighting,property_values_list, property_varied,round_dec)

        #ani_a =  multi_animation_weighting(fileName,data, cmap_weighting,  interval, fps, round_dec, nrows, ncols)
        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,property_varied_title,property_values_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,property_varied_title,property_values_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data,  cmap, interval, fps, round_dec, nrows, ncols,property_varied_title,property_values_list)
        #ani_e = live_compare_plot_animate_behaviour_scatter(fileName,data,norm_zero_one, cmap, nrows, ncols,property_varied, property_values_list,interval, fps,round_dec)

        if heterogenous_cultural_momentum:
            real_momentum_quick_list = [i.culture_momentum_quick_changers_real for i in data]
            real_momentum_lagards_list = [i.culture_momentum_lagards_real for i in data]

            live_print_heterogenous_culture_momentum(fileName, data, dpi_save, alpha_quick, alpha_normal, alpha_lagard, colour_quick, colour_normal, colour_lagard,nrows, ncols, property_varied_title, property_values_list, round_dec, real_momentum_quick_list, real_momentum_lagards_list)

        plt.show()



