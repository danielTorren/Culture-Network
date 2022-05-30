from network import Network
from utility import produceName,saveObjects,saveData,createFolder

def generate_data(time_steps_max, culture_var_min, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale):
    ### CREATE NETWORK
    #                         N, K, prob_wire, delta_t, M, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions
    social_network = Network( N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix,psi,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale)

    #### RUN TIME STEPS
    convergence = False
    time_counter = 0
    while time_counter < time_steps_max and convergence == False:
        social_network.next_step()
        time_counter += 1
        if social_network.cultural_var < culture_var_min:
            convergence = True

    return social_network

def run(time_steps_max, culture_var_min, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale):
    
    ###GENERATE THE DATA TO BE SAVED
    social_network = generate_data(time_steps_max, culture_var_min, N, K, prob_wire, delta_t, M, set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions,alpha_attract, beta_attract, alpha_threshold, beta_threshold,phi_list,learning_error_scale)
    steps = len(social_network.history_cultural_var)#bodge?
    #print(steps)
    #print(social_network.history_cultural_var)
    ###SAVE RUN DATA                N,  K, prob_wire, steps,behaviour_cap,delta_t,set_seed,M,culture_var_min,culture_div, nu, eta
    fileName = produceName(N, K, prob_wire, steps,delta_t,set_seed,M,culture_var_min,culture_div, nu, eta,alpha_attract, beta_attract, alpha_threshold, beta_threshold,learning_error_scale)
    dataName = createFolder(fileName)
    saveObjects(social_network, dataName)
    saveData(social_network, dataName,steps, N, M)
    print("File Path:", fileName)
    
    return fileName