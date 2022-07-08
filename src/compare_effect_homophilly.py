#compare the effect of varibale weighting on the model outcome

from run import generate_data
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA
import networkx as nx
from plot import ( 
    print_intial_culture_networks_homophily_fischer,
    print_culture_time_series_homophily_fischer,
    prints_culture_network_homophily_fischer
    )
from matplotlib.colors import LinearSegmentedColormap,  Normalize

# Params
reps = 9

save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False
linear_alpha_diff_state = False#if true use the exponential form instead like theo
homophily_state = True
alpha_change = True

#Social emissions model
K = 10  # k nearest neighbours INTEGER
M = 3  # number of behaviours
N = 100  # number of agents
total_time = 10.2

delta_t = 0.01  # time step size
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
learning_error_scale = 0.05  # 1 standard distribution is 2% error
carbon_emissions = [1]*M

#inverse_homophily = 0.1#0.2
homophilly_rate = 1.5

discount_factor = 0.6
present_discount_factor = 0.8

confirmation_bias = 1.5

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
    #"inverse_homophily": inverse_homophily,#1 is total mixing, 0 is no mixing
    "homophilly_rate": homophilly_rate,
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

dpi_save = 1200
layout = "circular"
cmap = LinearSegmentedColormap.from_list("BrownGreen", ["sienna", "white", "olivedrab"])
cmap_weighting = "Reds"
#norm_neg_pos = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1))
norm_neg_pos = Normalize(vmin=-1, vmax=1)
node_size = 50
nrows = 3
ncols = 3
layout = "circular"
round_dec = 2
frame_num = ncols * nrows - 1
bin_num = 1000

num_counts = 100000
frames_list = [int(round(x)) for x in np.linspace(0,time_steps_max, num=frame_num + 1)]
print("frames_list: ", frames_list)

if __name__ == "__main__":

        fileName = "results/fischer_homophilly_variation_%s_%s_%s_%s" % (str(params["N"]),str(params["time_steps_max"]),str(params["K"]), str(reps))
        data = []
        
        inverse_homophily_list = np.linspace(0,1,reps)

        #print(list(prob_rewire_list))
        for i in inverse_homophily_list:
            #print(i)
            params["inverse_homophily"] = i
            social_network = generate_data(params)
            data.append(social_network)

        createFolderSA(fileName)

        print_culture_time_series_homophily_fischer(fileName, data, dpi_save, nrows, ncols)
        print_intial_culture_networks_homophily_fischer(fileName, data, dpi_save, nrows, ncols , layout, norm_neg_pos, cmap, node_size)
        #prints_culture_network_homophily_fischer(fileName, data[0],layout, cmap ,node_size,  nrows, ncols ,  norm_neg_pos,frames_list, round_dec, dpi_save)
        #plt.show()