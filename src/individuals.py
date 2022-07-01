import numpy.typing as npt
import numpy as np

class Individual:

    """
    Class for indivduals
    """

    def __init__(
        self, init_data_attracts: npt.NDArray, init_data_thresholds: npt.NDArray, delta_t: float, culture_momentum: int, t: float, M: int, save_data: bool, carbon_intensive_list: list, discount_factor_list: list,
        attract_information_provision_list: list,
        nu: float,
        eta: float,
        t_IP_list: list,
    ):

        self.M = M
        self.t = t
        self.delta_t = delta_t
        self.save_data = save_data
        self.carbon_intensive_list = carbon_intensive_list
        self.culture_momentum = culture_momentum
        self.discount_factor_list = discount_factor_list
        self.attract_information_provision_list = attract_information_provision_list
        self.nu = nu
        self.eta = eta
        self.t_IP_list = t_IP_list
        self.init_thresholds = init_data_thresholds
        self.attracts, self.thresholds, self.values = self.create_behaviours(init_data_attracts, init_data_thresholds)
        
        self.information_provision = self.calc_information_provision()
        
        self.carbon_emissions, self.av_behaviour  = self.update_total_emissions_av_behaviour()
        self.av_behaviour_list = [self.av_behaviour]
        self.culture = self.calc_culture()

        if self.save_data:
            self.history_behaviour_values = [self.values]
            self.history_behaviour_attracts = [self.attracts]
            self.history_behaviour_thresholds = [self.thresholds]
            self.history_av_behaviour = [self.av_behaviour]
            self.history_culture = [self.culture]
            self.history_carbon_emissions = [self.carbon_emissions]
            self.history_information_provision = [self.information_provision]

    def create_behaviours(self, init_data_attracts: list, init_data_thresholds: list) -> tuple:
        return init_data_attracts, init_data_thresholds,init_data_attracts - init_data_thresholds


    def update_av_behaviour_list(self):
        #print("av behav: ",self.av_behaviour_list)
        if len(self.av_behaviour_list) < self.culture_momentum:
            self.av_behaviour_list.append(self.av_behaviour)
        else:
            self.av_behaviour_list.pop(0)
            self.av_behaviour_list.append(self.av_behaviour)

    def calc_culture(self) -> float:
        weighted_sum_behaviours = 0
        for i in range(len(self.av_behaviour_list)):
            weighted_sum_behaviours += self.discount_factor_list[i]*self.av_behaviour_list[i]
        normalized_culture = weighted_sum_behaviours/len(self.av_behaviour_list)
        #print(len(self.av_behaviour_list))
        return normalized_culture

    def update_values(self):
        self.values = self.attracts - self.thresholds

    def update_attracts(self,social_component_behaviours):
        #print("update attracts",social_component_behaviours,self.information_provision, type(self.information_provision))
        #print("before",self.attracts)
        self.attracts += self.delta_t*(social_component_behaviours + self.information_provision)  
        #print("after",self.attracts)

    def update_thresholds(self, carbon_price):

        for m in range(self.M):
            if self.init_thresholds[m] < carbon_price*self.carbon_intensive_list[m]:
                self.thresholds[m] = 0 
            else:
                self.thresholds[m] = self.init_thresholds[m] - carbon_price*self.carbon_intensive_list[m]

    def update_total_emissions_av_behaviour(self):
        total_emissions = 0  # calc_carbon_emission
        total_behaviour = 0  # calc_behaviour_av
        
        for i in range(self.M):

            total_behaviour += self.values[i]  # calc_behaviour_av

            if (self.values[i] <= 0):  # calc_carbon_emissions if less than or equal to 0 then it is a less environmetally friendly behaviour(brown)
                total_emissions += self.carbon_intensive_list[i]  # calc_carbon_emissions
        #print("total_behaviour:",total_behaviour)
        average_behaviour = total_behaviour/self.M
        return total_emissions, average_behaviour  # calc_carbon_emissions #calc_behaviour_a

    def calc_information_provision_boost(self,i):
        return self.attract_information_provision_list[i]*(1 - np.exp(-self.nu*(self.attract_information_provision_list[i] - self.attracts[i])))

    def calc_information_provision_decay(self,i):
        return self.information_provision[i]*np.exp(-self.eta*(self.t - self.t_IP_list[i]))

    def calc_information_provision(self):
        #print("time; ",self.t_IP_list)
        information_provision = []
        for i in range(self.M):
            if self.t_IP_list[i] == self.t:
                information_provision.append(self.calc_information_provision_boost(i))
            elif self.t_IP_list[i] < self.t and self.information_provision[i] > 0.00000001:
                information_provision.append(self.calc_information_provision_decay(i))
            else:
                information_provision.append(0) #this means that no information provision policy is ever present in this behaviour
        
        #print("information_provision:",information_provision)
        return np.array(information_provision)

    def update_information_provision(self):
        for i in range(self.M):
            if self.t_IP_list[i] == self.t:
                self.information_provision[i] = self.calc_information_provision_boost(i)
            elif self.t_IP_list[i] < self.t and self.information_provision[i] > 0.00000001:
                self.information_provision[i] = self.calc_information_provision_decay(i)
            else:
                self.information_provision[i] = 0 #this means that no information provision policy is ever present in this behaviour

    def save_data_individual(self):
        self.history_behaviour_values.append(self.values)
        self.history_behaviour_attracts.append(self.attracts)
        self.history_behaviour_thresholds.append(self.thresholds)
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_carbon_emissions.append(self.carbon_emissions)
        self.history_information_provision.append(self.information_provision)

    def next_step(self, t:float, social_component_behaviours: npt.NDArray, carbon_price:float):
        self.t = t
        self.update_information_provision()
        self.update_values()
        self.update_attracts(social_component_behaviours)
        self.update_thresholds(carbon_price)

        self.carbon_emissions, self.av_behaviour = self.update_total_emissions_av_behaviour()
        self.update_av_behaviour_list()
        self.culture = self.calc_culture()
        if self.save_data:
            self.save_data_individual()


