"""Performs sobol sensitivity analysis on the model. 

Created: 10/10/2022
"""

# imports
from package.plotting_data.sensitivity_analysis_plot import main

if __name__ == '__main__':
    main(
    fileName = "results/sensitivity_analysis_12_29_15__27_09_2023",
    plot_outputs = ["emissions_flow","var","emissions_flow_change"],
    dpi_save = 1200,
    latex_bool = 0,
    plot_dict = {
        "emissions_flow": {"title": r"$E/NM$", "colour": "red", "linestyle": "--"},
        "mu": {"title": r"$\mu_{I}$", "colour": "blue", "linestyle": "-"},
        "var": {"title": r"$\sigma^{2}_{I}$", "colour": "green", "linestyle": "*"},
        "coefficient_of_variance": {"title": r"$\sigma/\mu$","colour": "orange","linestyle": "-.",},
        "emissions_flow_change": {"title": r"$\Delta E/NM$", "colour": "brown", "linestyle": "-*"},
        "emissions_stock": {"title": r"$E_S/NM$", "colour": "black", "linestyle": "--"},
    },
    titles = [
        r"Number of individuals, $N$", 
        r"Number of green influencers, $N_G$", 
        r"Number of behaviours, $M$", 
        r"Mean neighbours, $K$",
        r"Cultural inertia, $\rho$",
        r"Social learning error, $ \sigma_{ \varepsilon}$ ",
        r"Initial attitude Beta, $a_A$",
        r"Initial attitude Beta, $b_A$",
        r"Initial threshold Beta, $a_T$",
        r"Initial threshold Beta, $b_T$",
        r"Discount factor, $\delta$",
        r"Attribute homophily, $h$",
        r"Confirmation bias, $\theta$"
    ]
)
