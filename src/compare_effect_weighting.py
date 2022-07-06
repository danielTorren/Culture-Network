#compare the effect of varibale weighting on the model outcome

from run import generate_data
import matplotlib.pyplot as plt
import numpy as np
from network import Network
from utility import createFolderSA

save_data = True
opinion_dynamics =  "DEGROOT" #  "DEGROOT"  "SELECT"

K = 2  # k nearest neighbours INTEGER
M = 3  # number of behaviours
N = 10  # number of agents
total_time = 1
delta_t = 0.01  # time step size

prob_rewire = 0.1  # re-wiring probability?

alpha_attract = 1  ##inital distribution parameters - doing the inverse inverts it!
beta_attract = 1
alpha_threshold = 8
beta_threshold = 2
time_steps_max = int(
    total_time / delta_t
)  # number of time steps max, will stop if culture converges
culture_momentum = 0.5# real time over which culture is calculated for INTEGER
set_seed = 1  ##reproducibility INTEGER
phi_list_lower,phi_list_upper = 0.1,1
learning_error_scale = 0.01  # 1 standard distribution is 2% error

#Infromation provision parameters
nu = 1# how rapidly extra gains in attractiveness are made
eta = 0.2#decay rate of information provision boost
attract_information_provision_list = np.array([0.5*(1/delta_t)]*M)#
t_IP_matrix = np.array([[],[],[]]) #REAL TIME; list stating at which time steps an information provision policy should be aplied for each behaviour

#Carbon price parameters
carbon_price_policy_start = 5#in simualation time to start the policy
carbon_price_init = 0.0#
#social_cost_carbon = 0.5
carbon_price_gradient = 0#social_cost_carbon/time_steps_max# social cost of carbon/total time
carbon_emissions = [1]*M

params = {
    "opinion_dynamics": opinion_dynamics,
    "save_data": save_data, 
    "time_steps_max": time_steps_max, 
    "delta_t": delta_t,
    "phi_list_lower": phi_list_lower,
    "phi_list_upper": phi_list_upper,
    "N": N,
    "M": M,
    "K": K,
    "prob_rewire": prob_rewire,
    "set_seed": set_seed,
    "culture_momentum": culture_momentum,
    "learning_error_scale": learning_error_scale,
    "alpha_attract": alpha_attract,
    "beta_attract": beta_attract,
    "alpha_threshold": alpha_threshold,
    "beta_threshold": beta_threshold,
    "nu":nu,
    "eta": eta,
    "attract_information_provision_list":attract_information_provision_list,
    "t_IP_matrix": t_IP_matrix,
    "carbon_price_init": carbon_price_init,
    "carbon_price_policy_start": carbon_price_policy_start,
    "carbon_price_gradient": carbon_price_gradient,
    "carbon_emissions" : carbon_emissions,
}

dpi_save = 1200


def plot_average_culture_comparison(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "Average Culture"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):
        #print(np.asarray(Data_list[i].history_average_culture))
        culture_min = np.asarray(Data_list[i].history_min_culture)  # bodge
        culture_max = np.asarray(Data_list[i].history_max_culture)  # bodge

        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_average_culture), label = i)
        ax.set_xlabel(r"Time")
        
        ax.fill_between(
            np.asarray(Data_list[i].history_time), culture_min, culture_max, alpha=0.5, linewidth=0
        )
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_av_cultures.png"
    fig.savefig(f, dpi=dpi_save)

def plot_carbon_emissions_total_comparison(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "Total Emissions"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):

        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_total_carbon_emissions), label = i)
        ax.set_xlabel(r"Time")
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_total_emissions.png"
    fig.savefig(f, dpi=dpi_save)

def plot_weighting_matrix_convergence_comparison(fileName: str, Data_list: list[Network], dpi_save:int):
    y_title = "weighting_matrix_convergence"

    fig, ax = plt.subplots()
    ax.set_ylabel(r"%s" % y_title)
    for i in range(len(Data_list)):

        ax.plot(np.asarray(Data_list[i].history_time), np.asarray(Data_list[i].history_weighting_matrix_convergence), label = i)
        ax.set_xlabel(r"Time")
    
    ax.legend()

    plotName = fileName + "/Plots"
    f = plotName + "/comparing_weighting_matrix_convergence.png"
    fig.savefig(f, dpi=dpi_save)

def plot_culture_timeseries(network: Network, title:str):
    y_title = "indivdiual culture"

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel(r"%s" % y_title)
    for i in network.agent_list:
        ax.plot(np.asarray(network.history_time ), np.asarray(i.history_culture))
        ax.set_xlabel(r"Time")


if __name__ == "__main__":

        fileName = "results/alpha_variation_%s_%s_%s" % (str(params["N"]),str(params["time_steps_max"]),str(params["K"]))

        #no change in attention
        params["alpha_change"] = 0
        social_network_case_zero = generate_data(params)
        #change in attention
        params["alpha_change"] = 1
        social_network_case_one = generate_data(params)
        #change in attention only once
        params["alpha_change"] = 0.5
        social_network_case_zero_point_five = generate_data(params)

        createFolderSA(fileName)

        data = [social_network_case_zero,social_network_case_one,social_network_case_zero_point_five]

        plot_average_culture_comparison(fileName, data, dpi_save)
        plot_carbon_emissions_total_comparison(fileName, data, dpi_save)
        plot_weighting_matrix_convergence_comparison(fileName, data, dpi_save)
        plot_culture_timeseries(social_network_case_zero, "zero")
        plot_culture_timeseries(social_network_case_one,"one")
        plot_culture_timeseries(social_network_case_zero_point_five,"zero point five")
        plt.show()



