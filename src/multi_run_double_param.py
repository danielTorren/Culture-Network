from logging import raiseExceptions
from run import parallel_run
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA,produce_param_list_double,generate_title_list
from matplotlib.colors import LinearSegmentedColormap,  Normalize
from matplotlib.cm import get_cmap
from plot import (
    live_print_culture_timeseries,
    print_culture_timeseries_vary_array,
    live_print_culture_timeseries_vary,
    live_compare_plot_animate_behaviour_scatter,
    live_phase_diagram_k_means_vary,
    print_culture_time_series_clusters_two_properties,
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
)

save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False#if true use the exponential form instead like theo
homophily_state = True
averaging_method = "Arithmetic"
harsh_data = False
heterogenous_cultural_momentum = False

compression_factor = 10

#Social emissions model
K = 10 # k nearest neighbours INTEGER
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
learning_error_scale = 0.02  # 1 standard distribution is 2% error
carbon_emissions = [1]*M

inverse_homophily = 1#0.2#0.2
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
    "averaging_method": averaging_method,
    "heterogenous_cultural_momentum" : heterogenous_cultural_momentum,
    "harsh_data": harsh_data,
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
    "alpha_change" : 1,
    "discount_factor": discount_factor,
    "inverse_homophily": inverse_homophily,#1 is total mixing, 0 is no mixing
    "homophilly_rate": homophilly_rate,
    "present_discount_factor": present_discount_factor,
    "confirmation_bias": confirmation_bias,
}

dpi_save = 1200
layout = "circular"
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "white", "olivedrab"])
cmap_weighting = "Reds"
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one = Normalize(vmin=0,vmax=1)
node_size = 50
bin_num = 1000
num_counts = 100000
fps = 5
interval = 50
round_dec = 2
cmap_edge = get_cmap("Greys")

min_k,max_k = 2,N - 1# Cover all the possible bases with the max k though it does slow it down
alpha_val = 0.25
size_points = 5
min_culture_distance = 0.5


if __name__ == "__main__":

        nrows = 2
        ncols = 2#due to screen ratio want more cols than rows usually
        reps = nrows*ncols
        if ncols*nrows > reps:
            raiseExceptions("Too many rows or columns for number of repetitions")

        param_min_col = 0.0
        param_max_col = 1.0#50.0
        param_min_row = 0.0
        param_max_row = 1.0#50.0

        property_row = "Confirmation bias"
        param_row = "confirmation_bias"
        property_col = "Inverse homophily"
        param_col = "inverse_homophily"


        fileName = "results/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (param_col,param_row,str(params["N"]),str(params["time_steps_max"]),str(params["K"]),str(param_min_col), str(param_max_col), str(param_min_row), str(param_max_row), str(reps))
        print("fileName: ", fileName)
        createFolderSA(fileName)

        ### GENERATE DATA
        row_list = np.linspace(param_min_row,param_max_row, nrows)
        col_list = np.linspace(param_min_col,param_max_col,ncols)
        params_list = produce_param_list_double(params,param_col,col_list,param_row,row_list)
        data_list  = parallel_run(params_list)  
        data_array = np.reshape(data_list, (len(row_list), len(col_list)))
        title_list = generate_title_list(property_col,col_list,property_row,row_list)


        ### PLOTS
        live_print_culture_timeseries_vary(fileName, data_list, property_row, property_col,title_list, nrows, ncols,  dpi_save)
        print_culture_timeseries_vary_array(fileName, data_array, property_col, col_list,property_row,row_list,  nrows, ncols , dpi_save)
        #live_phase_diagram_k_means_vary(fileName, data_array, property_row,  row_list,property_col,col_list,min_k,max_k,size_points, cmap_weighting,dpi_save)
        #print_culture_time_series_clusters_two_properties(fileName,data_array, row_list, col_list,property_row, property_col, min_k,max_k,size_points, alpha_val, min_culture_distance,"DTW", nrows, ncols, dpi_save, round_dec)

        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data_list,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,property_col,col_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data_list,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,property_col,col_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data_list,  cmap, interval, fps, round_dec, nrows, ncols,property_col,col_list)

        plt.show()


        plt.show()