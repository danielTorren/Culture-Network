from network import Network
from utility import produceName_alt, createFolder, saveObjects, saveData
import time


def generate_data(parameters: dict) -> Network:
    ### CREATE NETWORK
    ##params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale]
    #rint(parameters)
    #if parameters["save_data"]:
    start_time = time.time()
    # print("start_time =", time.ctime(time.time()))
    social_network = Network(parameters)
    
    #### RUN TIME STEPS
    time_counter = 0
    while time_counter < parameters["time_steps_max"]:
        social_network.next_step()
        time_counter += 1
    #if parameters["save_data"]:
    print(
        "SIMULATION time taken: %s minutes" % ((time.time() - start_time) / 60),
        "or %s s" % ((time.time() - start_time)),
    )
    return social_network


def run(parameters: dict, to_save_list: list, params_name: list) -> str:

    # params = [save_data,time_steps_max,M,N,phi_list,carbon_emissions,alpha_attract,beta_attract,alpha_threshold,beta_threshold,delta_t,K,prob_rewire,set_seed,culture_momentum,learning_error_scale]
    
    ###GENERATE THE DATA TO BE SAVED

    social_network = generate_data(parameters)
    #print("carbon emissions = ", social_network.total_carbon_emissions)
    if parameters["save_data"]:
        steps = len(social_network.history_cultural_var)
        start_time = time.time()
        fileName = produceName_alt(params_name)
        dataName = createFolder(fileName)
        saveObjects(social_network, dataName)
        saveData(
            social_network,
            dataName,
            steps,
            parameters["N"],
            parameters["M"],
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
    else:
        return 0
