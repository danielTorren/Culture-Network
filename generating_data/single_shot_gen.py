"""Runs a single simulation to produce data which is saved
A module that use dictionary of data for the simulation run. The single shot simualtion is run
for a given intial set seed.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 10/10/2022
"""
# imports
import time
import json
import numpy as np
from resources.run import generate_data
from resources.utility import (
    createFolder, 
    save_object, 
    load_object,
    produceName,
    produce_name_datetime
)

def main() -> str: 
    f = open("constants/base_params.json")
    base_params = json.load(f)
    base_params["time_steps_max"] = int(base_params["total_time"] / base_params["delta_t"])

    #fileName = produceName(params, params_name)
    root = "single_shot"
    fileName = produce_name_datetime(root)
    print("fileName:", fileName)

    Data = generate_data(base_params)  # run the simulation

    createFolder(fileName)
    save_object(Data, fileName + "/Data", "social_network")
    save_object(base_params, fileName + "/Data", "base_params")

    return fileName
