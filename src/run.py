from network import Network
from utility import produceName_alt, createFolder, saveObjects, saveData
import time


def generate_data(parameters: list) -> Network:
    ### CREATE NETWORK
    ##params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale]
    #rint(parameters)
    save_data = parameters[1]
    if save_data:
        start_time = time.time()
    # print("start_time =", time.ctime(time.time()))
    social_network = Network(parameters)
    time_steps_max = parameters[2]
    
    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < time_steps_max:
        social_network.next_step()
        time_counter += 1
    if save_data:
        print(
            "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
            "or %s s" % ((time.time() - start_time)),
        )
    return social_network


def run(parameters: list, to_save_list: list, params_name: list) -> str:

    # params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale]
    
    ###GENERATE THE DATA TO BE SAVED

    social_network = generate_data(parameters)
    #print("carbon emissions = ", social_network.total_carbon_emissions)
    steps = len(social_network.history_cultural_var)
    M = parameters[3]
    N = parameters[4]
    start_time = time.time()
    fileName = produceName_alt(params_name)
    dataName = createFolder(fileName)
    saveObjects(social_network, dataName)
    saveData(
        social_network,
        dataName,
        steps,
        N,
        M,
        to_save_list[0],
        to_save_list[1],
        to_save_list[2],
        to_save_list[3],
    )
    print("File Path:", fileName)
    print(
        "SAVE time taken: %s minutes" % ((time.time() - start_time) / 60),
        "or %s s" % ((time.time() - start_time)),
    )

    return fileName
