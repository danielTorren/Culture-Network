"""Contains functions that are not crucial to the simulation itself and are shared amongst files.
A module that aides in preparing folders, saving, loading and generating data for plots.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""

# imports
import pickle
import os
import numpy as np
from logging import raiseExceptions
from matplotlib.colors import Normalize, LogNorm
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import datetime
from sklearn.neighbors import KernelDensity

# modules
def createFolder(fileName: str) -> str:
    """
    Check if folders exist and if they dont create results folder in which place Data, Plots, Animations
    and Prints folders

    Parameters
    ----------
    fileName:
        name of file where results may be found

    Returns
    -------
    None
    """

    # print(fileName)
    # check for resutls folder
    if str(os.path.exists("results")) == "False":
        os.mkdir("results")

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


def save_object(data, fileName, objectName):
    """save single object as a pickle object

    Parameters
    ----------
    data: object,
        object to be saved
    fileName: str
        where to save it e.g in the results folder in data or plots folder
    objectName: str
        what name to give the saved object

    Returns
    -------
    None
    """
    with open(fileName + "/" + objectName + ".pkl", "wb") as f:
        pickle.dump(data, f)


def load_object(fileName, objectName) -> dict:
    """load single pickle file

    Parameters
    ----------
    fileName: str
        where to load it from e.g in the results folder in data folder
    objectName: str
        what name of the object to load is

    Returns
    -------
    data: object
        the pickle file loaded
    """
    with open(fileName + "/" + objectName + ".pkl", "rb") as f:
        data = pickle.load(f)
    return data


def produceName(parameters: dict, parameters_name_list: list) -> str:
    """produce a file name from a subset list of parameters and values  to create a unique identifier for each simulation run

    Parameters
    ----------
    params_dict: dict[dict],
        dictionary of parameters used to generate attributes, dict used for readability instead of super long list of input parameters.
        See generate_data function for an example
    parameters_name_list: list
        list of parameters to be used in the filename

    Returns
    -------
    fileName: str
        name of file where results may be found composed of value from the different assigned parameters.
    """

    fileName = "results/"
    for key, value in parameters.items():
        if key in parameters_name_list:
            fileName = fileName + "_" + str(key) + "_" + str(value)
    return fileName
    
def generate_vals_variable_parameters_and_norms(variable_parameters_dict):
    """using minimum and maximum values for the variation of a parameter generate a list of
     data and what type of distribution it uses

     Parameters
    ----------
    variable_parameters_dict: dict[dict]
        dictionary of dictionaries  with parameters used to generate attributes, dict used for readability instead of super
        long list of input parameters. Each key in this out dictionary gives the names of the parameter to be varied with details
        of the range and type of distribution of these values found in the value dictionary of each entry.

    Returns
    -------
    variable_parameters_dict: dict[dict]
        Same dictionary but now with extra entries of "vals" and "norm" in the subset dictionaries

    """
    for i in variable_parameters_dict.values():
        if i["divisions"] == "linear":
            i["vals"] = np.linspace(i["min"], i["max"], i["reps"])
            i["norm"] = Normalize()
        elif i["divisions"] == "log":
            i["vals"] = np.logspace(i["min"], i["max"], i["reps"])
            i["norm"] = LogNorm()
        else:
            raiseExceptions("Invalid divisions, try linear or log")
    return variable_parameters_dict
"""
def calc_num_clusters_specify_bandwidth(culture_data, s, bandwith):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(culture_data)
    e = kde.score_samples(s.reshape(-1,1))
    ma = argrelextrema(e, np.greater)[0]
    return len(ma)
"""

def calc_num_clusters_auto_bandwidth(culture_data, s):
    kde = gaussian_kde(culture_data)
    probs = kde.evaluate(s)
    ma_scipy = argrelextrema(probs, np.greater)[0]
    return len(ma_scipy)

def calc_num_clusters_set_bandwidth(culture_data,s,bandwidth):
    X_reshape = culture_data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_reshape)
    e = kde.score_samples(s.reshape(-1, 1))
    ma = argrelextrema(e, np.greater)[0]
    return len(ma)

def calc_pos_clusters_set_bandwidth(culture_data,s,bandwidth):
    X_reshape = culture_data.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X_reshape)
    e = kde.score_samples(s.reshape(-1,1))
    ma = argrelextrema(e, np.greater)[0]
    return ma


def produce_name_datetime(root):
    fileName = "results/" + root +  "_" + datetime.datetime.now().strftime("%H_%M_%S__%d_%m_%Y")
    return fileName