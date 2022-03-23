from behaviour import Behaviour 
from copy import deepcopy

class Individual():
    
    """ 
        Class for indivduals
        Properties: Culture, Behaviours list
    """
    def __init__(self, init_data_behaviours, delta_t):
        self.behaviours_list = self.create_behaviours(init_data_behaviours)
        self.len_behaviours_list = len(self.behaviours_list)
        self.culture = self.calc_culture()
        #print("HEEYY:",self.culture )
        self.identity = self.assign_idenity()

        self.delta_t = delta_t

        self.history_behaviours_list = [deepcopy(self.behaviours_list)]
        self.history_culture = [self.culture]

    def create_behaviours(self,init_data_behaviours):
        behaviours_list = []
        for i in range(len(init_data_behaviours)):
            behaviours_list.append(Behaviour(init_data_behaviours[i][0], init_data_behaviours[i][1], init_data_behaviours[i][2], init_data_behaviours[i][3], init_data_behaviours[i][4],init_data_behaviours[i][5]))
        return behaviours_list

    def calc_culture(self):
        total_culture = 0
        #print("out")
        for i in range(self.len_behaviours_list):
            if self.behaviours_list[i].behaviour_type == 1:
                total_culture += self.behaviours_list[i].value 
            else: 
                total_culture -= self.behaviours_list[i].value
            #print("inside culture count", total_culture)
        #the higher the cultural value the more it is pro envrionmental.

        av_culture = total_culture/self.len_behaviours_list
        #print("culture", av_culture)

        """
        if av_culture < -5:#cap the culture
            av_culture = -5
        elif av_culture > 5:
            av_culture = 5
        """

        return av_culture

    def assign_idenity(self):
        if self.culture < 0:
            self.identity = "anti_environmental"
        elif 0 <= self.culture < 0.5:
            self.identity = "indifferent"
        elif self.culture >= 0:
            self.identity = "pro_environmental"
        else:
            raise ValueError('invalid cultural values')
    
    def update_behaviours(self): # cultural_data, add in later the conformity bias
        for i in range(self.len_behaviours_list):
            self.behaviours_list[i].update_behaviour()
            
    """
    def update_dynamics(self,neighbourhood_data,cultural_data):
        # step 2

    """
        
    def update_attracts(self,social_component_behaviours):
        #step 3, equation 2

        """
        #OPTION A: weighted difference
        for i in range(self.len_behaviours_list):
            #print(self.behaviours_list[i].attract, self.delta_t*(social_component_behaviours[i] - self.behaviours_list[i].attract), social_component_behaviours[i] - self.behaviours_list[i].attract, social_component_behaviours[i] )
            self.behaviours_list[i].attract += self.delta_t*(social_component_behaviours[i] - self.behaviours_list[i].attract) # add in + IP + FA later
        """ 

        """
        #OPTION B: weighted average, but its time independent
        for i in range(self.len_behaviours_list):
            self.behaviours_list[i].attract = (self.behaviours_list[i].attract + social_component_behaviours[i])/2 # add in + IP + FA later
            #print(self.behaviours_list[i].attract)
        """


        """
        #OPTION C: weighted average time
        for i in range(self.len_behaviours_list):
            self.behaviours_list[i].attract += self.delta_t*((self.behaviours_list[i].attract + social_component_behaviours[i])/2 ) # add in + IP + FA later
        """

        """
        for i in range(self.len_behaviours_list):
            #print("attract before",self.behaviours_list[i].attract)
            self.behaviours_list[i].attract += self.delta_t*(social_component_behaviours[i]) # add in + IP + FA later
            #print("attract after",self.behaviours_list[i].attract)
        """

        #OPTION D: RK2 weighted difference,  NEED TO CHECK, it looks wrong
        for i in range(self.len_behaviours_list):
            self.behaviours_list[i].attract += self.delta_t*(self.delta_t/2 + 1)*( self.behaviours_list[i].attract  - social_component_behaviours[i]) # add in + IP + FA later
        #print("attract after",self.behaviours_list[i].attract)

    def update_costs(self):
        # step 3, equation 
        extras = 0 #for now do nothing
        for i in range(self.len_behaviours_list):
            self.behaviours_list[i].attract += self.delta_t*(extras) # add in + CP and TC

    def update_culture(self):
        #print("before",self.culture )
        self.culture = self.calc_culture()
        #print("after",self.culture )

    def save_data_individual(self):
        self.history_behaviours_list.append(deepcopy(self.behaviours_list))
        self.history_culture.append(self.culture)

    def next_step(self,social_component_behaviours):
        self.update_behaviours()#update the behaviours of agent
        #self.update_dynamics(neighbourhood_data,cultural_data) 
        self.update_attracts(social_component_behaviours)
        self.update_costs()
        self.update_culture()
        self.save_data_individual()