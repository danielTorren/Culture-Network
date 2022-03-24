import pickle
import csv
import os
import pandas as pd
import numpy as np

def produceName(P,  K, prob_wire, steps,behaviour_cap,delta_t,set_seed,Y,culture_var_min):
    runName = "network_" + str(P) + "_" + str(K) + "_" +  str(prob_wire) + "_" + str(steps) + "_" + str(behaviour_cap) +  "_" + str(delta_t) + "_" + str(set_seed) + "_" + str(Y) + "_" + str(culture_var_min)
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
    data_behaviour_cost = np.zeros([steps, P, Y])

    for i in range(steps):
        for j in range(P):
            for k in range(Y):
                data_behaviour_value[i][j][k] = data.agent_list[j].behaviour_list[k].history_value[i]
                data_behaviour_attract[i][j][k] = data.agent_list[j].behaviour_list[k].history_attract[i]
                data_behaviour_cost[i][j][k] = data.agent_list[j].behaviour_list[k].history_cost[i]

    dataDict["value"] = data_behaviour_value
    dataDict["attract"] = data_behaviour_attract
    dataDict["cost"] = data_behaviour_cost

    return dataDict

def save_list_array(dataName,dataSaveList,dataDict,agentClass):
    for i in dataSaveList:
        arrayName = dataName +  "/" + agentClass + "_" + i #'/Tenant_LandDet.csv'
        np.savez(arrayName,dataDict[i])


def saveData(data, dataName,steps, P, Y):
    
    ####TRANSPOSE STUFF SO THAT ITS AGGREGATED BY TIME STEP NOT INDIVIDUAL

    """save data from behaviours"""

    """
    flat_behaviour_data = []
    for i in range(len(data.agent_list)):
        for v in range(len(data.agent_list[i].behaviour_list)):
            flat_behaviour_data.append(data.agent_list[i].behaviour_list[v])#could i write this as a list comprehension

    #list of things to be saved
    data_save_behaviour_list = ["value","attract","cost"]
    #create dict with flatdata
    data_behaviour_dict = saveDataDict(flat_behaviour_data, data_save_behaviour_list)
    #TRANSPOSE
    data_behaviour_dict = transpose_data_dict(data_behaviour_dict,data_save_behaviour_list)
    """
    data_save_behaviour_list = ["value","attract","cost"]
    data_behaviour_dict = save_behaviours(data,steps, P, Y)
    #save as array
    save_list_array(dataName,data_save_behaviour_list,data_behaviour_dict,"behaviour")

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
    #list of things to be saved
    data_save_network_list = ["weighting_matrix","behavioural_attract_matrix","social_component_matrix"]
    #create dict with flatdata
    data_network_dict = saveDataDict([data],data_save_network_list)
    #save as array
    save_list_array(dataName,data_save_network_list,data_network_dict ,"network")

    #need ot save the average culture # IM NOT SAVING THE CULTURE AT THE MOMENT FUCK IT
    #saveFromList(dataName,["cultural_var"], {"cultural_var": data.history_cultural_var}, "network")

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




