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
    Class to represent green individuals with identities and behaviours

    [REDUCE THIS]
    ...

    Attributes
    ----------

    save_data : bool
        whether or not to save data. Set to 0 if only interested in end state of the simulation. If 1 will save
        data into timeseries (lists or lists of lists for arrays) which can then be either accessed directly in
        the social network object or saved into csv's
    compression_factor: int
        how often data is saved. If set to 1 its every step, then 10 is every 10th steps. Higher value gives lower
        resolution for graphs but more managable saved or end object size
    t: float
        keep track of time, increased with each step by time step delta_t
    delta_t: float
        size of time step in the simulation, default should be 1 and if this is changed then time dependant parameters
        such as phi(the degree of conspicuous consumption or social suseptability of a behaviour) also needs to be
        adjusted i.e. a larger delta_t requires a smaller phi to produce the same resolution of results
    M: int
        number of behaviours per individual. These behaviours are meant to represent action decisions that operate under the
        same identity such as the decision to cycle to work or take the car.
    carbon_emissions: list
        list of emissions of each behaviour, defaulted to 1 for each behaviour if its performed in a brown way, B =< 0
    phi_array: npt.NDArray[float]
        list of degree of social susceptibility or conspicous consumption of the different behaviours. Affects how much social interaction
        influences an individuals attitude towards a behaviour. As the behaviours are abstract the current values
        are not based on any empircal data hence the network behaviour is independent of their order so the
        list is generated using a linspace function using input parameters phi_array_lower and phi_array_upper. each element has domain = [0,1]
    values: npt.NDArray[float]
        array containing behavioural values, if greater than 0 then the green alternative behaviour is performed and emissions from that behaviour are 0. Domain =  [-1,1]
    av_behaviour
        mean attitude towards M behaviours at time t
    av_behaviour_list: list[float]
        time series of past average attitude values each given by av_behaviour, as far back as culture_momentum
    culture: float
        identity of the individual, if > 0.5 it is considered green. Determines who individuals pay attention to. Domain = [0,1]
    total_carbon_emissions: float
        total carbon emissions of that individual due to their behaviour
    history_behaviour_values: list[list[float]]
        timeseries of past behavioural values
    history_behaviour_attitudes: list[list[float]]
        timeseries of past behavioural attitudes
    self.history_behaviour_thresholds: list[list[float]]
        timeseries of past behavioural thresholds, static in the current model version
    self.history_av_behaviour: list[float]
        timeseries of past average behavioural attitudes
    self.history_culture: list[float]
        timeseries of past identity values
    self.history_carbon_emissions: list[float]
        timeseries of past individual total emissions

    Methods
    -------
    update_av_behaviour_list():
        Update moving average of past behaviours, inserting present value and 0th index and removing the oldest value
    calc_culture() -> float:
        Calculate the individual identity from past average attitudes weighted by the truncated quasi-hyperbolic discounting factor
    update_values():
        Update the behavioural values of an individual with the new attitudinal or threshold values
    update_attitudes(social_component):
        Update behavioural attitudes with social influence of neighbours mediated by the social susceptabilty of each behaviour phi
    calc_total_emissions():
        return total emissions of individual based on behavioural values
    save_data_individual():
        Save time series data
    next_step(t:float, steps:int, social_component: npt.NDArray):
        Push the individual simulation forwards one time step

    """

    def __init__(
        self,
        individual_params,
    ):
        """
        Constructs all the necessary attributes for the Green_influencer object.

        Parameters
        ----------
        individual_params: dict,
            useful parameters from the network

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
        Push the individual simulation forwards one time step. Update time, then behavioural values, attitudes and thresholds then calculate
        new identity of agent and save results.

        Parameters
        ----------
        steps: int
            Step counts in the simulation
        Returns
        -------
        None
        """
        self.t = t

        if self.save_timeseries_data and self.t % self.compression_factor == 0:
            self.save_data_individual()
