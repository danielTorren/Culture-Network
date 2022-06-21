import numpy.typing as npt

class Individual_ALT:

    """
    Class for indivduals
    Properties: Culture, Behaviours list
    """

    def __init__(
        self, init_data_attracts: npt.NDArray, init_data_thresholds: npt.NDArray, delta_t: float, culture_momentum: int, t: float, M: int, save_data: bool, carbon_intensive_list: list
    ):
        self.M = M
        self.t = t
        self.delta_t = delta_t
        self.save_data = save_data
        self.carbon_intensive_list = carbon_intensive_list
        self.attracts, self.thresholds, self.values = self.create_behaviours(init_data_attracts, init_data_thresholds)
        self.carbon_emissions = self.calc_individual_emissions()
        self.av_behaviour = self.calc_av_behaviour()
        self.av_behaviour_list = [self.av_behaviour]
        self.culture_momentum = culture_momentum
        self.culture = self.calc_culture()

        if self.save_data:
            self.history_behaviour_values = [self.values]
            self.history_behaviour_attracts = [self.attracts]
            self.history_behaviour_thresholds = [self.thresholds]
            self.history_av_behaviour = [self.av_behaviour]
            self.history_culture = [self.culture]
            self.history_carbon_emissions = [self.carbon_emissions]

    def create_behaviours(self, init_data_attracts: list, init_data_thresholds: list) -> tuple:
        return init_data_attracts, init_data_thresholds,init_data_attracts - init_data_thresholds

    def update_av_behaviour_list(self):
        if len(self.av_behaviour_list) < self.culture_momentum:
            self.av_behaviour_list.append(self.av_behaviour)
        else:
            self.av_behaviour_list.pop(0)
            self.av_behaviour_list.append(self.av_behaviour)

    def calc_culture(self) -> float:
        self.update_av_behaviour_list()
        return sum(self.av_behaviour_list)/ self.culture_momentum

    def update_values(self):
        self.values = self.attracts - self.thresholds

    def update_attract(self,social_component_behaviours):
        self.attracts += self.delta_t*(social_component_behaviours)  

    def calc_individual_emissions(self):
        return sum(self.carbon_intensive_list[i] for i in range(self.M) if self.values[i] <= 0)

    def calc_av_behaviour(self):
        return sum(self.values)/self.M

    def behaviours_next_step(self, social_component_behaviours: npt.NDArray) -> tuple[float, float]:
        self.update_values()
        self.update_attract(social_component_behaviours)

        return self.calc_individual_emissions(), self.calc_av_behaviour()
    
    def update_total_emissions_av_behaviour(self):
        total_emissions = 0  # calc_carbon_emission
        total_behaviour = 0  # calc_behaviour_av
        
        for i in range(self.M):
            total_behaviour += self.values[i]  # calc_behaviour_av
            if (self.values[i] <= 0):  # calc_carbon_emissions if less than or equal to 0 then it is a less environmetally friendly behaviour(brown)
                total_emissions += self.carbon_intensive_list[i]  # calc_carbon_emissions
        return total_emissions, total_behaviour / self.M  # calc_carbon_emissions #calc_behaviour_a

    def save_data_individual(self):
        self.history_behaviour_values.append(self.values)
        self.history_behaviour_attracts.append(self.attracts)
        self.history_behaviour_thresholds.append(self.thresholds)
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_carbon_emissions.append(self.carbon_emissions)

    def next_step(self, social_component_behaviours: npt.NDArray):
        self.update_values()
        self.update_attract(social_component_behaviours)
        self.carbon_emissions, self.av_behaviour = self.update_total_emissions_av_behaviour()
        self.culture = self.calc_culture()
        if self.save_data:
            self.save_data_individual()
