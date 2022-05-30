from network import Network
from utility import produceName,saveObjects,saveData,createFolder

def generate_data(time_steps_max, culture_var_min, P, K, prob_wire, delta_t, Y, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions):
    ### CREATE NETWORK
    #                         P, K, prob_wire, delta_t, Y, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions
    social_network = Network( P, K, prob_wire, delta_t, Y, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix,psi,carbon_price_init,carbon_price_gradient,carbon_emissions)

    #### RUN TIME STEPS
    convergence = False
    time_counter = 0
    while time_counter < time_steps_max and convergence == False:
        social_network.next_step()
        time_counter += 1
        if social_network.cultural_var < culture_var_min:
            convergence = True

    return social_network

def run(time_steps_max, culture_var_min, P, K, prob_wire, delta_t, Y, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions):
    
    ###GENERATE THE DATA TO BE SAVED
    social_network = generate_data(time_steps_max, culture_var_min, P, K, prob_wire, delta_t, Y, behaviour_cap,set_seed,culture_div,culture_momentum, nu, eta,attract_information_provision_list,t_IP_matrix ,psi,carbon_price_init,carbon_price_gradient,carbon_emissions)
    steps = len(social_network.history_cultural_var)#bodge?
    #print(steps)
    #print(social_network.history_cultural_var)
    ###SAVE RUN DATA                P,  K, prob_wire, steps,behaviour_cap,delta_t,set_seed,Y,culture_var_min,culture_div, nu, eta
    fileName = produceName(P, K, prob_wire, steps,behaviour_cap,delta_t,set_seed,Y,culture_var_min,culture_div, nu, eta)
    dataName = createFolder(fileName)
    saveObjects(social_network, dataName)
    saveData(social_network, dataName,steps, P, Y)
    print("File Path:", fileName)
    
    return fileName