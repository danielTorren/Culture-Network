import pickle
import csv
import os
import pandas as pd
import numpy as np
from network import Network
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as silhouette_score
from logging import raiseExceptions
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score as tslearn_silhouette_score
from matplotlib.colors import Normalize, LogNorm

def produceName_alt(parameters: list) -> str:
    fileName = "results/"
    for i in parameters:
        fileName = fileName + "_" + str(i)
    return fileName

def  produceName_SA(parameters_tuples: list) -> str:
    fileName = "results/SA"
    for i in parameters_tuples:
        fileName = fileName + "_" + str(i[0]) + "_" + str(i[1]) + "_" + str(i[2])
    return fileName

def  produceName_multi_run_n(variable_parameters_dict: list,fileName: str) -> str:
    for i in variable_parameters_dict.keys():
        fileName = fileName + "_" + i
    return fileName

def createFolder(fileName: str) -> str:
    # print(fileName)
    # check for resutls folder
    if str(os.path.exists("Results")) == "False":
        os.mkdir("Results")

    # check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    # make data folder:#
    dataName = fileName + "/Data"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)
    # make plots folder:
    plotsName = fileName + "/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make animation folder:
    plotsName = fileName + "/Animations"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make prints folder:
    plotsName = fileName + "/Prints"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    return dataName

def createFolderSA(fileName: str):
    # print(fileName)
    # check for resutls folder
    if str(os.path.exists("Results")) == "False":
        os.mkdir("Results")


    # check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    # make plots folder:
    plotsName = fileName + "/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make animation folder:
    plotsName = fileName + "/Animations"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    # make prints folder:
    plotsName = fileName + "/Prints"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)


def saveObjects(data: Network, name: str):
    with open(name + ".pkl", "wb") as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def loadObjects(name:str) -> Network:
    with open(name + ".pkl", "rb") as inp:
        data = pickle.load(inp)
    return data


def saveDataDict(flatData: list, dataSaveList: list) -> dict:
    dataDict = {}

    for i in dataSaveList:
        dataDict[i] = []

    for i in range(len(flatData)):
        for v in dataSaveList:
            dataDict[v].append(
                eval("flatData[i].history_" + v)
            )  # work out variable name and append this data usign eval to ge the variable
    #print("JJSJSJS",dataDict)

    return dataDict

def saveCSV(dataName: str, endName: str, dataSave: list):
    with open(dataName + endName, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(dataSave)


def saveFromList(dataName: str, dataSaveList: list, dataDict: dict, agentClass: str):
    for i in dataSaveList:
        endName = "/" + agentClass + "_" + i + ".csv"  #'/Tenant_LandDet.csv'
        saveCSV(dataName, endName, dataDict[i])

def save_behaviours(data: Network, steps: int, N: int, M: int) -> dict:
    #print("YO", type(data))
    # steps = steps + 1# include zeroth step BODGE!!!
    # print("steps",steps)
    dataDict = {}
    #print(steps, N, M)
    data_behaviour_value = np.zeros([steps, N, M])
    data_behaviour_attitude = np.zeros([steps, N, M])
    data_behaviour_threshold = np.zeros([steps, N, M])
    # print("steps",steps,N,M)
    for t in range(steps):
        #print("TIME")
        for n in range(N):
            for m in range(M):
                #print("value: ", data.agent_list[n].history_behaviour_values[t])
                data_behaviour_value[t][n][m] = (
                    data.agent_list[n].history_behaviour_values[t][m]
                )
                #print("ATRACT: ", data.agent_list[n].history_behaviour_attitudes[t])
                data_behaviour_attitude[t][n][m] = (
                    data.agent_list[n].history_behaviour_attitudes[t][m]
                )
                data_behaviour_threshold[t][n][m] = (
                    data.agent_list[n].history_behaviour_thresholds[t][m]
                )

    dataDict["value"] = data_behaviour_value
    dataDict["attitude"] = data_behaviour_attitude
    
    dataDict["threshold"] = data_behaviour_threshold

    # print(np.shape(dataDict["value"]))

    return dataDict


def save_list_array(dataName: str, dataSaveList: list, dataDict: dict, agentClass: str):
    for i in dataSaveList:
        arrayName = dataName + "/" + agentClass + "_" + i 
        np.savez(arrayName, dataDict[i])


def saveData(
    data: Network,
    dataName: str,
    steps: int,
    N: int,
    M: int,
    data_save_behaviour_array_list: list,
    data_save_individual_list: list,
    data_save_network_list: list,
    data_save_network_array_list: list,
):
    #print("HEEYY")

    ####TRANSPOSE STUFF SO THAT ITS AGGREGATED BY TIME STEP NOT INDIVIDUAL

    """save data from behaviours"""
    data_behaviour_array_dict = save_behaviours(data, steps, N, M)
    # save as array
    save_list_array(
        dataName, data_save_behaviour_array_list, data_behaviour_array_dict, "behaviour"
    )

    """save data from individuals"""
    # create dict with flatdata
    data_individual_dict = saveDataDict(data.agent_list, data_save_individual_list)
    # TRANSPOSE
    # data_individual_dict = transpose_data_dict(data_individual_dict,data_save_individual_list)
    # save as CSV
    saveFromList(
        dataName, data_save_individual_list, data_individual_dict, "individual"
    )

    """SAVE DATA FROM NETWORK"""
    # create dict with flatdata
    data_network_dict = saveDataDict([data], data_save_network_list)
    # save as CSV
    saveFromList(dataName, data_save_network_list, data_network_dict, "network")
    # create dict with flatdata
    data_network_array_dict = saveDataDict([data], data_save_network_array_list)
    # save as array
    save_list_array(
        dataName, data_save_network_array_list, data_network_array_dict, "network"
    )


def loadDataCSV(dataName: str, loadBoolean: list) -> dict:
    data = {}
    for i in loadBoolean:
        data[i] = pd.read_csv(dataName + "/" + i + ".csv", header=None)

    return data


def loadDataArray(dataName: str, loadBoolean: list) -> dict:
    data = {}
    for i in loadBoolean:
        a = np.load(dataName + "/" + i + ".npz", "r")

        b = [a[k] for k in a]
        c = b[0]

        if c.shape[0] == 1:  # SOOO FUCKING JANKY
            c = np.squeeze(c, axis=0)

        data[i] = c  # weird shit man

    return data


def loadData(dataName: str, loadBooleanCSV: list, loadBooleanArray: list) -> dict:
    dict_csv = loadDataCSV(dataName, loadBooleanCSV)
    dict_array = loadDataArray(dataName, loadBooleanArray)
    return dict_csv | dict_array  # merge the two dicts


def get_run_properties(Data: list, runName: str, paramList: list) -> dict:
    parameters = runName.split("_")
    parameters.pop(0)  # remove "network" and anything before

    for i in range(len(parameters)):
        try:
            eval(parameters[i])
            Data[paramList[i]] = float(parameters[i])
        except:
            Data[paramList[i]] = parameters[i]
    #print("SEKEET",type(Data))
    return Data


def frame_distribution(time_list:list , scale_factor: int, frames_proportion: float) -> list:
    #print("SEKEET",type(scale_factor))
    select = np.random.exponential(scale=scale_factor, size=frames_proportion)
    # print(select)
    norm_select = select / max(select)
    # print(norm_select)
    scaled_select = np.round(norm_select * (len(time_list) - 1))
    # print(scaled_select)
    frames_list = np.unique(scaled_select)
    # print(frames_list)
    frames_list_int = [int(x) for x in frames_list]
    print("frames:", frames_list_int)
    return frames_list_int


def frame_distribution_prints(time_list: list, scale_factor: int, frame_num: int):

    select = np.random.exponential(scale=scale_factor, size=frame_num)
    # print(select)
    norm_select = select / max(select)
    # print(norm_select)
    # print(norm_select)
    scaled_select = np.ceil(
        norm_select * (len(time_list) - 1)
    )  # stops issues with zero
    # print(np.ceil(norm_select*(len(time_list)-1)))
    # print(scaled_select)
    frames_list = list(np.unique(scaled_select))

    # print(type(frames_list),frames_list)

    while len(frames_list) < frame_num:
        select_in = np.random.exponential(scale=scale_factor, size=1)
        if select_in > max(select):
            continue# FIX THIS SO THAT IF A LARGER NUMEBR GETS PRODUCED IT TRIES AGAIN!
        else:
            norm_select_in = select_in / max(select)
            scaled_select = np.ceil(norm_select_in * (len(time_list) - 1))
            frames_list = list(frames_list + list(scaled_select))
            # print("in", select_in ,norm_select_in, scaled_select ,frames_list)

    # print(frames_list)
    frames_list_int = [int(x) for x in frames_list]
    # print(frames_list_int)
    frames_list_int.insert(0, 0)


    return sorted(frames_list_int)

def k_means_calc(Data,min_k,max_k,size_points):
    """
    #Z=pd.DataFrame(x) #converting into data frame for ease
    if size_points == 1:
        last_columns = np.asarray(Data["individual_culture"].iloc[: , -size_points]).reshape(-1,1)#THE RESHAPE MAKES IT AN ARRAY OF ARRAY WHERE EACH ENTRY HAS ITS own ENTRY INSTEAD OF A SINGLE LIST OF DATA
    elif size_points > Data["individual_culture"].shape[1]:
        raiseExceptions("Size points larger than number of available data points, lower")
    else:
        last_columns = np.asarray(Data["individual_culture"].iloc[: , -size_points:])
        print("Steps used: ",size_points,"/",len(Data["network_time"]), ", or last time taken",Data["network_time"][-size_points], "of ", Data["network_time"][-1])
    """

    last_columns = np.asarray(Data["individual_culture"])
    #print("Steps used: ",size_points,"/",len(time_list), ", or last time taken",time_list[-size_points], "of ", time_list[-1])

    scores = {}
    for k in range(min_k,max_k + 1):#+1 so it actually does the max number 
        tsKMean = TimeSeriesKMeans(n_clusters=k,
                            metric="dtw",
                            #metric_params={"gamma": .01},
                            verbose=False,
        )
        tsKMean.fit(last_columns)
        label=tsKMean.predict(last_columns)
        #print("last_columns",last_columns)
        #print("label",label)        
        #print("ts vs sk?", tslearn_silhouette_score(last_columns, label, metric="dtw"), silhouette_score(last_columns, label))
        #scores[k] = silhouette_score(last_columns, label)
        scores[k] = tslearn_silhouette_score(last_columns, label, metric="dtw")
    
    fin_max = max(scores, key=scores.get)

    print("k cluster, highest score = ",fin_max, scores[fin_max])

    return fin_max, scores[fin_max], scores

def live_k_means_calc(Data_culture, time_list,min_k,max_k,size_points):
    if size_points == 1:
        last_columns = Data_culture[:,-size_points:].reshape(-1,1)#THE RESHAPE MAKES IT AN ARRAY OF ARRAY WHERE EACH ENTRY HAS ITS own ENTRY INSTEAD OF A SINGLE LIST OF DATA
    elif size_points > Data_culture.shape[1]:
        print("Points difference = ",size_points,Data_culture.shape[1])
        raiseExceptions("Size points larger than number of available data points, lower")
    else:
        last_columns = Data_culture[:,-size_points:]
    #size_points = int(round(len(Data_culture[0])/2))
    #last_columns = Data_culture[:,-size_points:]
    print("Steps used: ",size_points,"/",len(time_list), ", or last time taken",time_list[-size_points], "of ", time_list[-1])
    #print(last_columns,last_columns.T)
    scores = {}
    for k in range(min_k,max_k + 1):#+1 so it actually does the max number 
        tsKMean_norm = TimeSeriesKMeans(n_clusters=k,
                            metric="dtw",
                            #metric_params={"gamma": .01},
                            verbose=False,
        )
        tsKMean_norm.fit(last_columns)
        label_norm=tsKMean_norm.predict(last_columns)
        scores[k] = tslearn_silhouette_score(last_columns, label_norm, metric="dtw")

        print(k,scores[k])
    
    fin_max = max(scores, key=scores.get)

    print("k cluster, highest score = ",fin_max, scores[fin_max])

    return fin_max, scores[fin_max], scores

def get_km_euclid(k_clusters,X_train):

    km = TimeSeriesKMeans(n_clusters=k_clusters, verbose=False)
    y_pred = km.fit_predict(X_train)#YOU HAVE TO FIT PREDICT TO THEN GET THE CLUSTER CENTERS LATER
    return km


def get_km_sDTW(k_clusters,X_train,gamma):
    sdtw_km = TimeSeriesKMeans(n_clusters=k_clusters,
                            metric="softdtw",
                            metric_params={"gamma": .01},
                            verbose=True,
    )

    y_pred = sdtw_km.fit_predict(X_train)
    return sdtw_km

def get_km_DTW(k_clusters,X_train,gamma):
    sdtw_km = TimeSeriesKMeans(n_clusters=k_clusters,
                            metric="dtw",
                            #metric_params={"gamma": .01},
                            verbose=True,
    )

    y_pred = sdtw_km.fit_predict(X_train)
    return sdtw_km

def produce_param_list(params,porperty_list, property):
    params_list = []
    for i in porperty_list:
        params[property] = i
        params_list.append(params.copy())#have to make a copy so that it actually appends a new dict and not just the location of the params dict
    return params_list

def add_varaiables_to_dict(params,variable_parameters_dict,X):
    for v in range(len(X)):
        params[variable_parameters_dict[v]["property"]] = X[v]
    return params

def produce_param_list_SA(param_values,params,variable_parameters_dict):
    "param_values are the satelli samples, params are the fixed variables, variable parameters is the list of SA variables, we want the name!"

    params_list = []
    for i, X in enumerate(param_values):
        variable_params_added = add_varaiables_to_dict(params,variable_parameters_dict,X)
        params_list.append(variable_params_added.copy())
    return params_list

def produce_param_list_double(params,param_col,col_list,param_row,row_list):
    params_list = []

    for i in row_list:
        for j in col_list:
            params[param_row] = i
            params[param_col] = j
            params_list.append(params.copy())
    return params_list

def produce_param_list_n(params,variable_parameters_dict):
    "create a list of the params. This only varies one parameter at a time with the other set to whatever is in the params dict"
    params_list = []
    for i in variable_parameters_dict.values():
        for j in range(i["reps"]):
            params_copy = params.copy()#Copy it so that i am varying one parameter at a time independently
            params_copy[i["property"]] = i["vals"][j]
            params_list.append(params_copy)
    return params_list

def produce_param_list_n_double(params,variable_parameters_dict, param_row,param_col):
    "create a list of the params. This only varies one parameter at a time with the other set to whatever is in the params dict"
    params_list = []

    for i in variable_parameters_dict[param_row]["vals"]:
        for j in variable_parameters_dict[param_col]["vals"]:
            params[param_row] = i
            params[param_col] = j
            params_list.append(params.copy())

    print("params_list",params_list)
    
    return params_list

def generate_title_list(
    property_col,
    col_list,
    property_row,
    row_list,
    round_dec,
    ):

    title_list = []
    
    for i in range(len(row_list)):
        for j in range(len(col_list)):
            title_list.append(("%s = %s, %s = %s") % (property_row,str(round(row_list[i],round_dec)), property_col,str(round(col_list[j], round_dec))))

    print(title_list)
    
    return  title_list

def sa_save_Y(Y,fileName, YName):
    with open(fileName + "/%s.pkl" % (YName), 'wb') as f:
        pickle.dump(Y, f)
    
def sa_load_Y(fileName, YName) -> dict:
    with open(fileName + "/%s.pkl" % (YName), 'rb') as f:
        Y = pickle.load(f)
    return Y

def sa_save_problem(problem,fileName):
    with open(fileName + "/problem.pkl", 'wb') as f:
        pickle.dump(problem, f)
    
def sa_load_problem(fileName) -> dict:
    with open(fileName + "/problem.pkl", 'rb') as f:
        problem = pickle.load(f)
    return problem


def multi_n_save_data_list(data_list,fileName):
    with open(fileName + "/data_list.pkl", 'wb') as f:
        pickle.dump(data_list, f)
    
def multi_n_load_data_list(fileName) -> dict:
    with open(fileName + "/data_list.pkl", 'rb') as f:
        data_list = pickle.load(f)
    return data_list


def multi_n_save_mean_data_list(mean_data_list,fileName):
    with open(fileName + "/mean_data_list.pkl", 'wb') as f:
        pickle.dump(mean_data_list, f)
    
def multi_n_load_mean_data_list(fileName) -> dict:
    with open(fileName + "/mean_data_list.pkl", 'rb') as f:
        mean_data_list = pickle.load(f)
    return mean_data_list

def multi_n_save_coefficient_variance_data_list(coefficient_variance_data_list,fileName):
    with open(fileName + "/coefficient_variance_data_list.pkl", 'wb') as f:
        pickle.dump(coefficient_variance_data_list, f)
    
def multi_n_load_coefficient_variance_data_list(fileName) -> dict:
    with open(fileName + "/coefficient_variance_data_list.pkl", 'rb') as f:
        coefficient_variance_data_list = pickle.load(f)
    return coefficient_variance_data_list


def multi_n_save_variable_parameters_dict_list(variable_parameters_dict,fileName):
    with open(fileName + "/variable_parameters_dict.pkl", 'wb') as f:
        pickle.dump(variable_parameters_dict, f)
    
def multi_n_load_variable_parameters_dict_list(fileName) -> dict:
    with open(fileName + "/variable_parameters_dict.pkl", 'rb') as f:
        variable_parameters_dict = pickle.load(f)
    return variable_parameters_dict

def multi_n_save_combined_data(combined_data,fileName):
    with open(fileName + "/combined_data.pkl", 'wb') as f:
        pickle.dump(combined_data, f)
    
def multi_n_load_combined_data(fileName) -> dict:
    with open(fileName + "/combined_data.pkl", 'rb') as f:
        combined_data = pickle.load(f)
    return combined_data


def generate_vals_variable_parameters_and_norms(variable_parameters_dict):
    for i in variable_parameters_dict.values():
        if i["divisions"] == "linear": 
            i["vals"] = np.linspace(i["min"],i["max"],i["reps"])
            i["norm"] = Normalize()
        elif i["divisions"] == "log": 
            i["vals"] = np.logspace(i["min"],i["max"], i["reps"])
            i["norm"] = LogNorm()
        else:
            raiseExceptions("Invalid divisions, try linear or log")
    return variable_parameters_dict





