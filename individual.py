from turtle import update
from behaviour import Behaviour 
import numpy as np

class Individual():
    
    """ 
        Class for indivduals
        Properties: Culture, Behaviours list
    """
    def __init__(self, init_data_behaviours, delta_t,culture_momentum,attract_information_provision_list,nu,eta,t,t_IP_list):
        self.behaviour_list = self.create_behaviours(init_data_behaviours)
        self.Y = len(self.behaviour_list)
        self.av_behaviour = self.calc_behaviour_av()
        self.av_behaviour_list = [self.av_behaviour]
        self.culture_momentum = culture_momentum
        self.culture = self.calc_culture()

        self.t = t
        self.delta_t = delta_t

        self.attract_information_provision_list = attract_information_provision_list
        self.nu = nu
        self.eta = eta
        self.t_IP_list = t_IP_list
        self.information_provision = self.calc_information_provision()

        self.history_av_behaviour = [self.av_behaviour]
        self.history_culture = [self.culture]

    def create_behaviours(self,init_data_behaviours):
        behaviour_list = []
        for i in range(len(init_data_behaviours)):
            #init_attract, init_threshold,behaviour_cap, carbon_emissions,attract_individual_learning,psi
            #print("attrct individ learn",init_data_behaviours[i][4])
            behaviour_list.append(Behaviour(init_data_behaviours[i][0], init_data_behaviours[i][1], init_data_behaviours[i][2], init_data_behaviours[i][3],init_data_behaviours[i][4],init_data_behaviours[i][5]))
        return behaviour_list

    def calc_behaviour_av(self):
        total_behaviour = 0
        for i in range(self.Y):
            #print(self.behaviour_list[i].value)
            total_behaviour += self.behaviour_list[i].value 
        av_behaviour = total_behaviour/self.Y
        #print("av_behaviour", av_behaviour)
        return av_behaviour
        #the higher the cultural value the more it is pro envrionmental.
    
    def update_av_behaviour(self):
        self.av_behaviour = self.calc_behaviour_av()

    def update_av_behaviour_list(self):
        if len(self.av_behaviour_list) < self.culture_momentum:
            self.av_behaviour_list.append(self.av_behaviour)
        else:
            self.av_behaviour_list.pop(0)
            self.av_behaviour_list.append(self.av_behaviour)

    def calc_culture(self):
        total_culture = 0
        for i in self.av_behaviour_list:
                total_culture += i
        av_culture = total_culture/self.culture_momentum
        #print("culture", av_culture)
        return av_culture
    
    def update_behaviours(self): # cultural_data, add in later the conformity bias
        for i in range(self.Y):
            self.behaviour_list[i].next_step()

    def calc_information_provision_boost(self,i):
        return self.attract_information_provision_list[i]*(1 - np.exp(-self.nu*(self.attract_information_provision_list[i] - self.behaviour_list[i].attract)))

    def calc_information_provision_decay(self,i):
        return self.information_provision[i]*np.exp(-self.eta*(self.t - self.t_IP_list[i]))

    def calc_information_provision(self):
        information_provision = []
        for i in range(self.Y):
            if self.t_IP_list[i] == self.t:
                information_provision.append(self.calc_information_provision_boost(i))
            elif self.t_IP_list[i] < self.t:
                information_provision.append(self.calc_information_provision_decay(i))
            else:
                information_provision.append(0) #this means that no information provision policy is ever present in this behaviour
        return information_provision

    def update_information_provision(self):
        self.information_provision = self.calc_information_provision()

    def update_attracts(self,social_component_behaviours,attract_cultural_group):
        for i in range(self.Y):
            self.behaviour_list[i].update_attract(social_component_behaviours[i],attract_cultural_group[i],self.information_provision[i],self.delta_t)

        #print("attract after",self.behaviour_list[i].attract)

    def update_thresholds(self, carbon_price_gradient):
        for i in range(self.Y):
            self.behaviour_list[i].update_threshold(carbon_price_gradient)

    def update_culture(self):
        #print("before",self.culture )
        self.culture = self.calc_culture()
        #print("after",self.culture )

    def save_data_individual(self):
        #self.history_behaviour_list.append(deepcopy(self.behaviour_list))
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)

    def next_step(self,social_component_behaviours,attract_cultural_group,carbon_price_gradient):
        self.update_information_provision()
        self.update_behaviours()#update the behaviours of agent
        self.update_attracts(social_component_behaviours,attract_cultural_group)
        self.update_thresholds(carbon_price_gradient)
        self.update_av_behaviour()
        self.update_av_behaviour_list()
        self.update_culture()
        
        self.save_data_individual()