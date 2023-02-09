# imports
import json
import numpy as np
from resources.utility import produce_name_datetime, createFolder,save_object
from resources.run import parallel_run 
from generating_data.oneD_param_sweep_gen import (
    produce_param_list,
)

def main() -> str: 

    property_varied = "alpha_change"
    title_list = [r"Static uniform $\alpha_{n,k}$", r"Static culturally determined $\alpha_{n,k}$", r"Dynamic culturally determined $\alpha_{n,k}$"]
    property_values_list = ["A", "B", "C"]

    f = open("constants/base_params.json")
    base_params = json.load(f)

    root = "alpha_change_micro_consensus_single"
    fileName = produce_name_datetime(root)
    print("fileName: ", fileName)

    params_list = produce_param_list(base_params, np.asarray(property_values_list), property_varied)

    data_list = parallel_run(params_list)
    createFolder(fileName)

    save_object(data_list, fileName + "/Data", "data_list")
    save_object(base_params, fileName + "/Data", "base_params")
    save_object(property_varied, fileName + "/Data", "property_varied")
    save_object(title_list, fileName + "/Data", "title_list")
    save_object(property_values_list, fileName + "/Data", "property_values_list")

    return fileName

