import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from utility import loadData, get_run_properties


def get_jenks_breaks(data_list, number_class):
    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_class + 1):
        kclass.append(min(data_list))
    kclass[number_class] = float(data_list[len(data_list) - 1])
    count_num = number_class
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass


###
#JENKS
"""
breaks = get_jenks_breaks(x, 5)

for line in breaks:
    plt.plot([line for _ in range(len(x))], 'k--')

plt.plot(x)
plt.grid(True)
plt.show()
"""


#Y= 2 + np.random.rand(50,2)
#Z= np.concatenate((X,Y))




###LOAD DATA
#x= np.random.rand(50,1)

loadBooleanCSV = [
    "individual_culture",
]  
loadBooleanArray = []
paramList = [
    "opinion_dynamics",
    "time_steps_max",
    "M",
    "N",
    "delta_t",
    "K",
    "prob_rewire",
    "set_seed",
    "learning_error_scale",
    "alpha_attract",
    "beta_attract",
    "alpha_threshold",
    "beta_threshold",
    "culture_momentum_real",
]
# "network_cultural_var",,"network_carbon_price"

FILENAME = "results/_DEGROOT_1000_3_100_0.1_10_0.1_1_0.02_1_1_1_1_1"
dataName = FILENAME + "/Data"
Data = loadData(dataName, loadBooleanCSV, loadBooleanArray)
Data = get_run_properties(Data, FILENAME, paramList)

#print(Data["individual_culture"])

x = np.asarray(Data["individual_culture"])[-1]

#a,b,c = k_means(x)

print(a,b,c)

