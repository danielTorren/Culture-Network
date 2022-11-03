from resources.network import Network
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

params = {
    "total_time": 2000,
    "delta_t": 1.0,
    "compression_factor": 10,
    "save_data": 1, 
    "alpha_change" : 1.0,
    "degroot_aggregation": 1,
    "averaging_method": "Arithmetic",
    "phi_lower": 0.005,
    "phi_upper": 0.01,
    "N": 200,
    "M": 5,
    "K": 20,
    "set_seed": 1,
    "seed_list": [1,2,3,4,5],
    "culture_momentum_real": 10,
    "learning_error_scale": 0.02,
    "discount_factor": 0.95,
    "confirmation_bias": 30,
    "homophilly_rate" : 1,
    "a_attitude": 0.5,
    "b_attitude": 0.5,
    "a_threshold": 1,
    "b_threshold": 1,
    "action_observation": 0.0,
    "green_N": 0,
    "network_structure": "small_world"
}

if __name__ == "__main__":

    f = "results/homophily_generation"
    dpi_save = 1200

    homophily_list = np.linspace(0, 1, 11)
    homophily_list_label = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    homophily_list_pos = [0,2,4,6,8, 10]

    prob_rewire_list = np.linspace(0, 1, 11)
    prob_rewire_list_label =  ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    prob_rewire_list_pos = [0,2,4,6,8, 10]

    total_difference_init_matrix = []

    for i in homophily_list:
        params["homophily"] = i
        total_difference_init_row = []
        for j in prob_rewire_list:
            params["prob_rewire"] = j
            social_network = Network(params)
            total_difference_init_row.append(sum(social_network.total_identity_differences)/(params["N"]))
        total_difference_init_matrix.append(total_difference_init_row)

    fig, ax = plt.subplots()
    
    mat = ax.matshow(
        total_difference_init_matrix,
        aspect="auto",
        cmap=get_cmap("Reds"),
    )
    cbar = fig.colorbar(
        mat,
        #plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=Z.min(), vmax=Z.max())),
        ax=ax,
    )
    cbar.set_label("Total identity difference between neighbours, per agent")

    ax.set_xticks(prob_rewire_list_pos) 
    ax.set_xticklabels(prob_rewire_list_label, fontsize=12)
    ax.xaxis.set_ticks_position('bottom')

    ax.set_yticks(homophily_list_pos) 
    ax.set_yticklabels(homophily_list_label, fontsize=12)

    ax.set_xlabel(r"Probability of rewire")
    ax.set_ylabel(r"Attribute homophily")

    fig.savefig(f + "Total_difference_between_neighbours_per_agent"+ ".png", dpi=dpi_save, format="png")

    plt.show()






