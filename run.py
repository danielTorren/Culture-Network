from network import Network
from utility import produceName,saveObjects,saveData,createFolder

def generate_data(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max,culture_var_min):
    ### CREATE NETWORK
    social_network = Network( P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed)

    #### RUN TIME STEPS
    convergence = False
    time_counter = 0
    while time_counter < time_steps_max and convergence == False:
        social_network.next_step()
        time_counter += 1
        if social_network.cultural_var < culture_var_min:
            convergence = True

    return social_network

def run(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max,culture_var_min):
    
    ###GENERATE THE DATA TO BE SAVED
    social_network = generate_data(P,  K, prob_wire, delta_t, Y, name_list, behave_type_list, behaviour_cap,set_seed,time_steps_max,culture_var_min)
    steps = len(social_network.history_cultural_var)#len(social_network.history_cultural_var) - 1
    #print(social_network.history_cultural_var)
    ###SAVE RUN DATA
    fileName, runName = produceName(P,  K, prob_wire, steps,behaviour_cap,delta_t,set_seed,Y,culture_var_min)
    dataName = createFolder(fileName)
    saveObjects(social_network, dataName)
    saveData(social_network, dataName,steps, P, Y)
    print("File Path:", fileName)
    
    return fileName
