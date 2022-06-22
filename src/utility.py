import pickle
import csv
import os
import pandas as pd
import numpy as np
from network import Network

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
    data_behaviour_attract = np.zeros([steps, N, M])
    data_behaviour_threshold = np.zeros([steps, N, M])
    data_behaviour_information_provision = np.zeros([steps, N, M])
    #print("steps",steps,N,M)
    for t in range(steps):
        #print("TIME")
        for n in range(N):
            for m in range(M):
                #print("value: ", data.agent_list[n].history_behaviour_values[t])
                data_behaviour_value[t][n][m] = (
                    data.agent_list[n].history_behaviour_values[t][m]
                )
                #print("ATRACT: ", data.agent_list[n].history_behaviour_attracts[t])
                data_behaviour_attract[t][n][m] = (
                    data.agent_list[n].history_behaviour_attracts[t][m]
                )
                data_behaviour_threshold[t][n][m] = (
                    data.agent_list[n].history_behaviour_thresholds[t][m]
                )
                data_behaviour_information_provision[t][n][m] = (
                    data.agent_list[n].history_information_provision[t][m]
                )

    dataDict["value"] = data_behaviour_value
    dataDict["attract"] = data_behaviour_attract
    
    dataDict["threshold"] = data_behaviour_threshold
    dataDict["information_provision"] = data_behaviour_information_provision

    # print(np.shape(dataDict["value"]))

    return dataDict


def save_list_array(dataName: str, dataSaveList: list, dataDict: dict, agentClass: str):
    for i in dataSaveList:
        arrayName = dataName + "/" + agentClass + "_" + i  #'/Tenant_LandDet.csv'
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
    # print("frames prints:",sorted(frames_list_int))

    return sorted(frames_list_int)
