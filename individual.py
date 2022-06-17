from behaviour import Behaviour 

class Individual():
    
    """ 
        Class for indivduals
        Properties: Culture, Behaviours list
    """
    def __init__(self, init_data_behaviours, delta_t,culture_momentum,t,M,save_data):
        self.M = M
        self.t = t
        self.delta_t = delta_t
        self.save_data = save_data

        self.behaviour_list = self.create_behaviours(init_data_behaviours)

        self.carbon_emissions = self.init_calc_carbon_emissions()
        self.av_behaviour = self.init_calc_behaviour_av()
        self.av_behaviour_list = [self.av_behaviour]
        self.culture_momentum = culture_momentum
        self.culture = self.calc_culture()

        if self.save_data:
            self.history_av_behaviour = [self.av_behaviour]
            self.history_culture = [self.culture]
            self.history_carbon_emissions = [self.carbon_emissions]

    def create_behaviours(self,init_data_behaviours):
        #init_attract, init_threshold,carbon_emissions
        behaviour_list = [Behaviour(init_data_behaviours[i][0], init_data_behaviours[i][1], init_data_behaviours[i][2],init_data_behaviours[i][3], self.delta_t,self.save_data) for i in range(len(init_data_behaviours))]
        return behaviour_list

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
    """
    def update_behaviours(self): # cultural_data, add in later the conformity bias
        for i in range(self.M):
            self.behaviour_list[i].next_step()

    def update_attracts(self,social_component_behaviours):#,attract_cultural_group,attract_cultural_group[i]
        for i in range(self.M):
            self.behaviour_list[i].update_attract(social_component_behaviours[i])
    """

    def init_calc_carbon_emissions(self):
        total_emissions = 0
        for i in range(self.M):
            if self.behaviour_list[i].value > 0:
                total_emissions += self.behaviour_list[i].carbon_emissions 
        return total_emissions 

    def init_calc_behaviour_av(self):
        total_behaviour = 0
        for i in range(self.M):
            #print(self.behaviour_list[i].value)
            total_behaviour += self.behaviour_list[i].value 
        av_behaviour = total_behaviour/self.M

        return av_behaviour
    

    def behaviours_next_step(self,social_component_behaviours):
        total_emissions = 0#calc_carbon_emissions
        total_behaviour = 0#calc_behaviour_av
        for i in range(self.M):
            self.behaviour_list[i].next_step()#update_behaviours
            self.behaviour_list[i].update_attract(social_component_behaviours[i])#update_attracts
            
            total_behaviour += self.behaviour_list[i].value#calc_behaviour_av

            if self.behaviour_list[i].value > 0:#calc_carbon_emissions
                total_emissions += self.behaviour_list[i].carbon_emissions #calc_carbon_emissions

        av_behaviour = total_behaviour/self.M#calc_behaviour_av
        return total_emissions ,av_behaviour#calc_carbon_emissions #calc_behaviour_av

    def save_data_individual(self):
        #self.history_behaviour_list.append(deepcopy(self.behaviour_list))
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_carbon_emissions.append(self.carbon_emissions)

    def next_step(self,social_component_behaviours):
        self.carbon_emissions,self.av_behaviour = self.behaviours_next_step(social_component_behaviours)
        #self.update_behaviours()#update the behaviours of agent
        #self.update_attracts(social_component_behaviours)#,attract_cultural_group
        #self.av_behaviour = self.calc_behaviour_av()
        self.update_av_behaviour_list()
        self.culture = self.calc_culture()
        #self.carbon_emissions = self.calc_carbon_emissions()
        if self.save_data:
            self.save_data_individual()