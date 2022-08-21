#compare the effect of varibale weighting on the model outcome

from run import generate_data
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from matplotlib.cm import get_cmap
from utility import createFolderSA
from plot import (
    plot_average_culture_comparison,
    plot_carbon_emissions_total_comparison,
    plot_weighting_matrix_convergence_comparison,
    plot_cum_weighting_matrix_convergence_comparison,
    print_culture_timeseries,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    print_culture_time_series_clusters,
)

from matplotlib.colors import LinearSegmentedColormap,  Normalize

# Params
save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False#if true use the exponential form instead like theo
homophily_state = True
#alpha_change = True

compression_factor = 5

#Social emissions model
K = 10  # k nearest neighbours INTEGER
M = 3  # number of behaviours
N = 100  # number of agents
total_time = 20

delta_t = 0.05  # time step size
culture_momentum_real = 1# real time over which culture is calculated for INTEGER, NEEDS TO BE MROE THAN DELTA t

prob_rewire = 0.2  # re-wiring probability?

alpha_attract = 1#2  ##inital distribution parameters - doing the inverse inverts it!
beta_attract = 1#3
alpha_threshold = 1#3
beta_threshold = 1#2

time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges

set_seed = 1  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 0.1,1
learning_error_scale = 0.02 # 1 standard distribution is 2% error
carbon_emissions = [1]*M

inverse_homophily = 0.2#0.2
homophilly_rate = 1

discount_factor = 0.6
present_discount_factor = 0.8

confirmation_bias = 10

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
    #"alpha_change" : alpha_change,
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

fps = 5
interval = 50
round_dec = 2
cmap_edge = get_cmap("Greys")

layout = "circular"
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "white", "olivedrab"])
cmap_weighting = "Reds"
#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one  = Normalize(vmin=0, vmax=1)
node_size = 50
bin_num = 1000
dpi_save = 1200

min_k,max_k = 2,N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5

if __name__ == "__main__":

        fileName = "results/alpha_variation_%s_%s_%s" % (str(params["N"]),str(params["time_steps_max"]),str(params["K"]))

        #no change in attention
        params["alpha_change"] = 0
        social_network_case_zero = generate_data(params)
        #change in attention only once
        params["alpha_change"] = 0.5
        social_network_case_zero_point_five = generate_data(params)
        #change in attention
        params["alpha_change"] = 1
        social_network_case_one = generate_data(params)

        createFolderSA(fileName)

        data = [social_network_case_zero,social_network_case_zero_point_five,social_network_case_one]

        title_list = ["Equal Weighting", "Static Cultural Weighting", "Dynamic Cultural Weighting"]
        nrows = 1
        ncols = 3

        #plot_average_culture_comparison(fileName, data, dpi_save,title_list )
        #plot_carbon_emissions_total_comparison(fileName, data, dpi_save, title_list)
        plot_weighting_matrix_convergence_comparison(fileName, data, dpi_save, title_list)
        plot_cum_weighting_matrix_convergence_comparison(fileName, data, dpi_save, title_list)

        #print_culture_timeseries(fileName, data , title_list, nrows, ncols ,dpi_save)
        print_culture_time_series_clusters(fileName, data, title_list, "Link", min_k,max_k,size_points, alpha_val, min_culture_distance, nrows, ncols, dpi_save, round_dec)

        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,"Weighting state",title_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,"Weighting state",title_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data,  cmap, interval, fps, round_dec, nrows, ncols,"Weighting state",title_list)

        plt.show()



