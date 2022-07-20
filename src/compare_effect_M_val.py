#compare the effect of varibale weighting on the model outcome

from run import generate_data
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA
from matplotlib.colors import LinearSegmentedColormap,  Normalize
from matplotlib.cm import get_cmap
from plot import (
    live_compare_animate_culture_network_and_weighting,
    live_compare_animate_behaviour_matrix,
    live_compare_animate_weighting_matrix,
    live_compare_animate_behaviour_matrix,
    plot_average_culture_comparison,
    plot_carbon_emissions_total_comparison,
    plot_weighting_matrix_convergence_comparison,
    print_culture_timeseries,
)


# Params
save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False#if true use the exponential form instead like theo
homophily_state = True
alpha_change = True
nur_attitude = False
value_culture_def = False
harsh_data = False

#Social emissions model
K = 10 # k nearest neighbours INTEGER
#M = 3  # number of behaviours
N = 100  # number of agents

total_time = 100

culture_momentum_real = 5# real time over which culture is calculated for INTEGER, NEEDS TO BE MROE THAN DELTA t
delta_t = 0.1  # time step size

prob_rewire = 0.1  # re-wiring probability?

time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges

set_seed = 1  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 1,1

#print(phi_list)
learning_error_scale = 0.01  # 1 standard distribution is 2% error

inverse_homophily = 0.3#0.2
homophilly_rate = 1

discount_factor = 0.6
present_discount_factor = 0.8

confirmation_bias = 100

#harsh data parameters
if harsh_data:
    green_extreme_max = 8
    green_extreme_min = 2
    green_extreme_prop = 2/5
    indifferent_max = 2
    indifferent_min = 2
    indifferent_prop = 1/5
    brown_extreme_min = 2
    brown_extreme_max = 8
    brown_extreme_prop = 2/5
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
    "time_steps_max": time_steps_max, 
    "carbon_price_state" : carbon_price_state,
    "information_provision_state" : information_provision_state,
    "linear_alpha_diff_state": linear_alpha_diff_state,
    "homophily_state": homophily_state,
    "alpha_change" : alpha_change,
    "delta_t": delta_t,
    "phi_list_lower": phi_list_lower,
    "phi_list_upper": phi_list_upper,
    "N": N,
    #"M": M,
    "K": K,
    "prob_rewire": prob_rewire,
    "set_seed": set_seed,
    "culture_momentum_real": culture_momentum_real,
    "learning_error_scale": learning_error_scale,
    #"carbon_emissions" : carbon_emissions,
    "discount_factor": discount_factor,
    "inverse_homophily": inverse_homophily,#1 is total mixing, 0 is no mixing
    "homophilly_rate" : homophilly_rate,
    "present_discount_factor": present_discount_factor,
    "confirmation_bias": confirmation_bias,
    "nur_attitude": nur_attitude,
    "value_culture_def": value_culture_def, 
    "harsh_data": harsh_data,
    
}
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

dpi_save = 1200
layout = "circular"
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "white", "olivedrab"])
cmap_weighting = "Reds"
#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
norm_zero_one = Normalize(vmin=0,vmax=1)
node_size = 50
bin_num = 1000
num_counts = 100000
fps = 5
interval = 50
round_dec = 2
cmap_edge = get_cmap("Greys")


if __name__ == "__main__":

        M_val_init = 2
        M_val_max_minus_one = 10

        fileName = "results/m_variation_%s_%s_%s_%s_%s" % (str(params["N"]),str(params["time_steps_max"]),str(params["K"]), str(M_val_init), str(M_val_max_minus_one))
        print("fileName: ", fileName)

        nrows = 2
        ncols = 3

        M_val_list = [1,2,4,6,8,10]#np.arange(M_val_init,M_val_max_minus_one, 1)

        print("M_val_list: ", M_val_list)

        data = []
        for i in M_val_list:
            params["M"] = i
            params["carbon_emissions"] = [1]*i

            res = generate_data(params)
            data.append(res)
            #print("RES:", res.history_weighting_matrix[0])
            #print("RES LATER",res.history_weighting_matrix[-1])

        createFolderSA(fileName)

        #plot_carbon_emissions_total_M_val(fileName, data, dpi_save)
        #plot_weighting_convergence_M_val(fileName, data, dpi_save)
        #print_culture_time_series_M_val(fileName, data, dpi_save, nrows, ncols)
        #print_intial_culture_networks_M_val(fileName, data, dpi_save, nrows, ncols , layout, norm_zero_one, cmap, node_size)
        #prints_init_weighting_matrix_M_val(fileName, data, dpi_save,nrows, ncols, cmap_weighting)
        #prints_final_weighting_matrix_M_val(fileName, data, dpi_save,nrows, ncols, cmap_weighting)

        plot_average_culture_comparison(fileName, data, dpi_save,M_val_list )
        plot_carbon_emissions_total_comparison(fileName, data, dpi_save, M_val_list)
        plot_weighting_matrix_convergence_comparison(fileName, data, dpi_save, M_val_list)
        print_culture_timeseries(fileName, data , M_val_list, nrows, ncols ,dpi_save)

        #multi_animation_weighting(fileName,data, cmap_weighting,  interval, fps, round_dec, nrows, ncols, time_steps_max)
        #ani_b = live_compare_animate_culture_network_and_weighting(fileName,data,layout,cmap,node_size,interval,fps,norm_zero_one,round_dec,cmap_edge, ncols, nrows,"Number of behaviours",M_val_list)
        #ani_c = live_compare_animate_weighting_matrix(fileName, data,  cmap_weighting, interval, fps, round_dec, cmap_edge, nrows, ncols,"Number of behaviours",M_val_list)
        #ani_d = live_compare_animate_behaviour_matrix(fileName, data,  cmap, interval, fps, round_dec, nrows, ncols,"Number of behaviours",M_val_list)
        
        plt.show()



