#compare the effect of varibale weighting on the model outcome

from run import generate_data
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA
import networkx as nx
from plot import ( 
    prod_pos,
    plot_carbon_emissions_total_prob_rewire,
    plot_weighting_convergence_prob_rewire,
    print_culture_time_series_prob_rewire,
    print_intial_culture_networks_prob_rewire,
    plot_beta_distributions,
    prints_init_weighting_matrix_prob_rewire,
    )
from matplotlib.colors import LinearSegmentedColormap,  Normalize

save_data = True
opinion_dynamics =  "SELECT" #  "DEGROOT"  "SELECT"
carbon_price_state = False
information_provision_state = False

#Social emissions model
K = 15  # k nearest neighbours INTEGER
M = 3  # number of behaviours
N = 50  # number of agents
total_time = 20
delta_t = 0.01  # time step size
#prob_rewire = 0.2  # re-wiring probability?
alpha_attract = 0.2#2  ##inital distribution parameters - doing the inverse inverts it!
beta_attract = 0.2#3
alpha_threshold = 0.2#3
beta_threshold = 0.2#2
time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges
culture_momentum = 0.1# real time over which culture is calculated for INTEGER
culture_momentum_steps = round(culture_momentum/ delta_t)
set_seed = 10  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 0.1,1
learning_error_scale = 0.01  # 1 standard distribution is 2% error
carbon_emissions = [1]*M

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
    "delta_t": delta_t,
    "phi_list_lower": phi_list_lower,
    "phi_list_upper": phi_list_upper,
    "N": N,
    "M": M,
    "K": K,
    #"prob_rewire": prob_rewire,
    "set_seed": set_seed,
    "culture_momentum": culture_momentum,
    "learning_error_scale": learning_error_scale,
    "alpha_attract": alpha_attract,
    "beta_attract": beta_attract,
    "alpha_threshold": alpha_threshold,
    "beta_threshold": beta_threshold,
    "carbon_emissions" : carbon_emissions,
    "alpha_change" : 1,
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

bin_num = 1000
num_counts = 100000


if __name__ == "__main__":

        fileName = "results/homophilly_variation_%s_%s_%s" % (str(params["N"]),str(params["time_steps_max"]),str(params["K"]))
        data = []
        reps = 9
        nrows = 3
        ncols = 3
        
        prob_rewire_list = np.linspace(0,1,reps)

        #print(list(prob_rewire_list))
        for i in prob_rewire_list:
            #print(i)
            params["prob_rewire"] = i
            social_network = generate_data(params)
            data.append(social_network)

        createFolderSA(fileName)

        plot_beta_distributions(fileName,alpha_attract,beta_attract,alpha_threshold,beta_threshold,bin_num,num_counts,dpi_save)

        plot_carbon_emissions_total_prob_rewire(fileName, data, dpi_save,culture_momentum)
        plot_weighting_convergence_prob_rewire(fileName, data, dpi_save,culture_momentum)
        print_culture_time_series_prob_rewire(fileName, data, dpi_save, nrows, ncols,culture_momentum)
        print_intial_culture_networks_prob_rewire(fileName, data, dpi_save, nrows, ncols , layout, norm_neg_pos, cmap, node_size)
        prints_init_weighting_matrix_prob_rewire(fileName, data, dpi_save,nrows, ncols, cmap_weighting)

        plt.show()