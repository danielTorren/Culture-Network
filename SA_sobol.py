from run import generate_data
from utility import produceName_random,createFolder
import time
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

if __name__ == "__main__":

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359]]
    }

    M = 3#number of behaviours
    N = 50#number of agents
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
    attract_information_provision_list = [1]*M#
    t_IP_matrix = [[],[],[],[],[],[]] #list of lists stating at which time steps an information provision policy should be aplied for each behaviour

    #Individual learning rate
    psi = 1#

    #Carbon price parameters
    carbon_price_init = 0#
    social_cost_carbon = 0
    carbon_price_gradient = social_cost_carbon/time_steps_max# social cost of carbon/total time
    carbon_emissions = [0.3,0.6,0.8]#np.random.random_sample(M)#[1]*M# these should based on some paramters



    #GENERATE PARAMETER VALUES
    param_values = saltelli.sample(problem, 1024)#NumPy matrix. #N(2D +2) samples where N is 1024 and D is the number of parameters
    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
        social_network = generate_data(time_steps_max, culture_var_min, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_policy_start,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale)

        Y[i] = social_network.evaluate_model(X)

    #run the thing
    social_network = generate_data(time_steps_max, culture_var_min, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_policy_start,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale)

    #GET RESULTS FOR EACH RUN AND PUT INTO TXT
    Y = np.loadtxt("outputs.txt", float)# I NEED TO LOAD IN THE RESUTLS

    ## ANALYZE RESULTS
    Si = sobol.analyze(problem, Y) # Si is a Python dict with the keys "S1", "S2", "ST", "S1_conf", "S2_conf", and "ST_conf". The _conf keys store the corresponding confidence intervals, typically with a confidence level of 95%.

    #Visualize
    Si.plot()