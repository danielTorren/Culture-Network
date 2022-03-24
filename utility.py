import pickle
import csv
import os
import pandas as pd

def produceName(steps, P,  K, prob_wire, delta_t, Y,behaviour_cap,set_seed):
    runName = "network_" + str(P) + "_" + str(K) + "_" +  str(prob_wire) + "_" + str(steps) + "_" + str(behaviour_cap) +  "_" + str(delta_t) + "_" + str(set_seed) + "_" + str(Y)
    fileName = "Results/"+ runName
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

def saveCSV(dataName,endName,dataSave):
    with open(dataName + endName, 'w', newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(dataSave)

def saveFromList(dataName,dataSaveList,dataDict,agentClass):
    for i in dataSaveList:
        endName = "/" + agentClass + "_" + i + ".csv"  #'/Tenant_LandDet.csv'
        saveCSV(dataName,endName,dataDict[i])

def saveData(data, dataName):
    
    """save data from behaviours"""
    flat_behaviour_data = []
    for i in range(len(data.agent_list)):
        for v in range(len(data.agent_list[i].behaviour_list)):
            flat_behaviour_data.append(data.agent_list[i].behaviour_list[v])#could i write this as a list comprehension

    #list of things to be saved
    data_save_behaviour_list = ["value","attract","cost"]
    #create dict with flatdata
    data_behaviour_dict = saveDataDict(flat_behaviour_data, data_save_behaviour_list)
    #save as CSV
    saveFromList(dataName,data_save_behaviour_list,data_behaviour_dict,"behaviour")

    """save data from individuals"""
    #list of things to be saved
    data_save_individual_list = ["culture"]
    #create dict with flatdata
    data_individual_dict = saveDataDict(data.agent_list,data_save_individual_list)
    #save as CSV
    saveFromList(dataName,data_save_individual_list,data_individual_dict,"individual")

    """SAVE DATA FROM NETWORK"""
    #list of things to be saved
    data_save_network_list = ["weighting_matrix","behavioural_attract_matrix","social_component_matrix","cultural_var"]
    #create dict with flatdata
    data_network_dict = saveDataDict([data],data_save_network_list)
    #save as CSV
    saveFromList(dataName,data_save_network_list,data_network_dict ,"network")

def loadData(dataName, loadBoolean):
    data = {}
    for i in loadBoolean:
        data[i] = pd.read_csv(dataName + "/" + i +".csv",header=None )

    return data

def get_run_properties(Data,runName,paramList):
    parameters = runName.split("_")
    parameters.pop(0)#remove "network" and anything before
    print("parameters",parameters)

    for i in range(len(parameters)):
        Data[paramList[i]] = float(parameters[i])

    return Data




