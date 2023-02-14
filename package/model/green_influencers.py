"""Define Green_influencer agent class
A module that defines "individuals" that have vectors of attitudes towards behaviours which dont evolve and are always green.

Author: Daniel Torren Peraire Daniel.Torren@uab.cat dtorrenp@hotmail.com

Created: 26/10/2022
"""

# imports
import numpy as np
import numpy.typing as npt

# modules
class Green_influencer:

    """
    Class to represent green individuals with identities and behaviours that dont evolve

    """

    def __init__(
        self,
        individual_params,
    ):
        """
        Constructs all the necessary attributes for the Green_influencer object.

        """

        self.M = individual_params["M"]
        self.save_timeseries_data = individual_params["save_timeseries_data"]
        self.compression_factor = individual_params["compression_factor"]

        self.attitudes = np.asarray(self.M*[1.00])
        self.thresholds = np.asarray(self.M*[0.00])
        self.av_behaviour = 1.00
        self.values = self.attitudes - self.thresholds
        self.culture = 1.00
        self.total_carbon_emissions,self.behavioural_carbon_emissions = self.calc_total_emissions()
        self.green_fountain_state = 1
        self.attitudes_star = np.asarray(self.M*[1.00])

        if self.save_timeseries_data:
            self.history_behaviour_values = [list(self.values)]
            self.history_behaviour_attitudes = [list(self.attitudes)]
            self.history_behaviour_thresholds = [list(self.thresholds)]
            self.history_av_behaviour = [self.av_behaviour]
            self.history_culture = [self.culture]
            self.history_carbon_emissions = [self.total_carbon_emissions]
            self.history_behavioural_carbon_emissions = [self.behavioural_carbon_emissions]

    def calc_total_emissions(self):
        """
        Return total emissions of individual based on behavioural values

        Parameters
        ----------
        None

        Returns
        -------
        float
        """
        behavioural_emissions = [((1-self.values[i])/2) for i in range(self.M)]
        return sum(behavioural_emissions),behavioural_emissions

    def save_data_individual(self):
        """
        Save time series data

        Parameters
        ----------neigh
        None

        Returns
        -------
        None
        """
        self.history_behaviour_values.append(list(self.values))
        self.history_behaviour_attitudes.append(list(self.attitudes))
        self.history_behaviour_thresholds.append(list(self.thresholds))
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_carbon_emissions.append(self.total_carbon_emissions)
        self.history_behavioural_carbon_emissions.append(self.behavioural_carbon_emissions)

    def next_step(self, t: float, social_component: npt.NDArray):
        """
        Push the individual simulation forwards one time step.
        """
        self.t = t

        if self.save_timeseries_data and self.t % self.compression_factor == 0:
            self.save_data_individual()
