from behaviour import Behaviour 

class Individual():
    
    """ 
        Class for indivduals
        Properties: Culture, Behaviours list
    """
    def __init__(self, init_data_behaviours, delta_t,culture_momentum,t,M):
        self.M = M
        self.t = t
        self.delta_t = delta_t

        self.behaviour_list = self.create_behaviours(init_data_behaviours)
        
        self.av_behaviour = self.calc_behaviour_av()
        self.av_behaviour_list = [self.av_behaviour]
        self.culture_momentum = culture_momentum
        self.culture = self.calc_culture()
        self.carbon_emissions = self.calc_carbon_emissions()

    def create_behaviours(self,init_data_behaviours):
        #init_attract, init_threshold,carbon_emissions
        behaviour_list = [Behaviour(init_data_behaviours[i][0], init_data_behaviours[i][1], init_data_behaviours[i][2], self.delta_t) for i in range(len(init_data_behaviours))]
        return behaviour_list

    def calc_behaviour_av(self):
        total_behaviour = 0
        for i in range(self.M):
            #print(self.behaviour_list[i].value)
            total_behaviour += self.behaviour_list[i].value 
        av_behaviour = total_behaviour/self.M

        return av_behaviour

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

        return av_culture
    
    def update_behaviours(self): # cultural_data, add in later the conformity bias
        for i in range(self.M):
            self.behaviour_list[i].next_step()

    def update_attracts(self,social_component_behaviours):#,attract_cultural_group,attract_cultural_group[i]
        for i in range(self.M):
            self.behaviour_list[i].update_attract(social_component_behaviours[i])

    def calc_carbon_emissions(self):
        total_emissions = 0
        for i in range(self.M):
            if self.behaviour_list[i].value > 0:
                total_emissions += self.behaviour_list[i].carbon_emissions 
        return total_emissions 

    def next_step(self,social_component_behaviours):
        self.update_behaviours()#update the behaviours of agent
        self.update_attracts(social_component_behaviours)#,attract_cultural_group
        self.av_behaviour = self.calc_behaviour_av()
        self.update_av_behaviour_list()
        self.culture = self.calc_culture()
        self.carbon_emissions = self.calc_carbon_emissions()