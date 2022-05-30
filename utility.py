import pickle
import csv
import os
import pandas as pd
import numpy as np

def produceName(P,  K, prob_wire, steps,behaviour_cap,delta_t,set_seed,Y,culture_var_min,culture_div,nu, eta):
    runName = "network_" + str(P) + "_" + str(K) + "_" +  str(prob_wire) + "_" + str(steps) + "_" + str(behaviour_cap) +  "_" + str(delta_t) + "_" + str(set_seed) + "_" + str(Y) + "_" + str(culture_var_min) + "_" + str(culture_div) + "_" + str(nu) + "_" + str(eta) 
    fileName = "results/"+ runName
    return fileName, runName

def createFolder(fileName):
    #check for resutls folder
    if str(os.path.exists("Results")) == "False":
        os.mkdir("Results")

    #check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    #make data folder:#
    dataName = fileName +"/Data"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)
    #make plots folder:
    plotsName = fileName +"/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    #make animation folder:
    plotsName = fileName +"/Animations"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    #make prints folder:
    plotsName = fileName +"/Prints"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    return dataName

def saveObjects(data,name):
    with open(name +'.pkl', 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def loadObjects(name):
    with open(name +'.pkl', 'rb') as inp:
        data = pickle.load(inp)
    return data

def saveDataDict(flatData,dataSaveList):
    dataDict = {}

    for i in dataSaveList:
        dataDict[i] = []

    for i in range(len(flatData)):
        for v in dataSaveList:
            dataDict[v].append(eval("flatData[i].history_" + v ))#work out variable name and append this data usign eval to ge the variable
    return dataDict

def transpose_data_dict(data,dataSaveList):
    for i in range(len(dataSaveList)):
        data[dataSaveList[i]] = (np.asarray(data[dataSaveList[i]])).transpose()
    return data


def saveCSV(dataName,endName,dataSave):
    with open(dataName + endName, 'w', newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(dataSave)

def saveFromList(dataName,dataSaveList,dataDict,agentClass):
    for i in dataSaveList:
        endName = "/" + agentClass + "_" + i + ".csv"  #'/Tenant_LandDet.csv'
        saveCSV(dataName,endName,dataDict[i])

def save_behaviours(data,steps, P, Y):
    dataDict = {}
    data_behaviour_value = np.zeros([steps, P, Y])
    data_behaviour_attract = np.zeros([steps, P, Y])
    data_behaviour_threshold = np.zeros([steps, P, Y])

    for i in range(steps):
        for j in range(P):
            for k in range(Y):
                data_behaviour_value[i][j][k] = data.agent_list[j].behaviour_list[k].history_value[i]
                data_behaviour_attract[i][j][k] = data.agent_list[j].behaviour_list[k].history_attract[i]
                data_behaviour_threshold[i][j][k] = data.agent_list[j].behaviour_list[k].history_threshold[i]

    dataDict["value"] = data_behaviour_value
    dataDict["attract"] = data_behaviour_attract
    dataDict["threshold"] = data_behaviour_threshold

    return dataDict

def save_list_array(dataName,dataSaveList,dataDict,agentClass):
    for i in dataSaveList:
        arrayName = dataName +  "/" + agentClass + "_" + i #'/Tenant_LandDet.csv'
        np.savez(arrayName,dataDict[i])


def saveData(data, dataName,steps, P, Y):
    
    ####TRANSPOSE STUFF SO THAT ITS AGGREGATED BY TIME STEP NOT INDIVIDUAL

    """save data from behaviours"""
    
    data_save_behaviour_array_list = ["value","attract","threshold"]
    data_behaviour_array_dict = save_behaviours(data,steps, P, Y)
    #save as array
    save_list_array(dataName,data_save_behaviour_array_list,data_behaviour_array_dict,"behaviour")

    """save data from individuals"""
    #list of things to be saved
    data_save_individual_list = ["culture"]
    #create dict with flatdata
    data_individual_dict = saveDataDict(data.agent_list,data_save_individual_list)
    #TRANSPOSE
    #data_individual_dict = transpose_data_dict(data_individual_dict,data_save_individual_list)
    #save as CSV
    saveFromList(dataName,data_save_individual_list,data_individual_dict,"individual")
    
    """SAVE DATA FROM NETWORK"""

    data_save_network_list = ["time","cultural_var","carbon_price"]
    #create dict with flatdata
    data_network_dict = saveDataDict([data],data_save_network_list)
    #save as CSV
    saveFromList(dataName,data_save_network_list,data_network_dict,"network")

    #list of things to be saved
    data_save_network_array_list = ["weighting_matrix","behavioural_attract_matrix","social_component_matrix"]
    #create dict with flatdata
    data_network_array_dict = saveDataDict([data],data_save_network_array_list)
    #save as array
    save_list_array(dataName,data_save_network_array_list,data_network_array_dict ,"network")

def loadDataCSV(dataName, loadBoolean):
    data = {}
    for i in loadBoolean:
        data[i] = pd.read_csv(dataName + "/" + i +".csv",header=None )

    return data


def loadDataArray(dataName, loadBoolean):
    data = {}
    for i in loadBoolean:
        a = np.load(dataName + "/" + i + ".npz","r")

        b = [a[k] for k in a]
        c = b[0]

        if c.shape[0] == 1:#SOOO FUCKING JANKY
            c = np.squeeze(c, axis=0)

        data[i] = c#weird shit man

    return data

def loadData(dataName, loadBooleanCSV,loadBooleanArray):
    dict_csv = loadDataCSV(dataName, loadBooleanCSV)
    dict_array = loadDataArray(dataName, loadBooleanArray)
    return  dict_csv | dict_array#merge the two dicts


def get_run_properties(Data,runName,paramList):
    parameters = runName.split("_")
    parameters.pop(0)#remove "network" and anything before
    #print("parameters",parameters)

    for i in range(len(parameters)):
        Data[paramList[i]] = float(parameters[i])

    return Data

def frame_distribution(time_list,scale_factor,frames_proportion):
    select = np.random.exponential(scale = scale_factor, size = frames_proportion)
    #print(select)
    norm_select = select/max(select)
    #print(norm_select)
    scaled_select = np.round(norm_select*(len(time_list)-1))
    #print(scaled_select)
    frames_list = np.unique(scaled_select)
    #print(frames_list)
    frames_list_int = [int(x) for x in frames_list]
    #print(frames_list_int)
    return frames_list_int

def frame_distribution_prints(time_list,scale_factor,frame_num):

    select = np.random.exponential(scale = scale_factor, size = frame_num)
    #print(select)
    norm_select = select/max(select)
    #print(norm_select)
    #print(norm_select)
    scaled_select = np.ceil(norm_select*(len(time_list)-1))#stops issues with zero
    #print(np.ceil(norm_select*(len(time_list)-1)))
    #print(scaled_select)
    frames_list = np.unique(scaled_select)
    #print(frames_list)
    frames_list_int = [int(x) for x in frames_list]
    #print(frames_list_int)
    frames_list_int.insert(0,0)
    
    return frames_list_int




