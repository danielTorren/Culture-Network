from behaviour import Behaviour 

class Individual():
    
    """ 
        Class for indivduals
        Properties: Culture, Behaviours list
    """
    def __init__(self, init_data_behaviours, delta_t):
        self.behaviour_list = self.create_behaviours(init_data_behaviours)
        self.len_behaviour_list = len(self.behaviour_list)
        self.culture = self.calc_culture()
        #print("HEEYY:",self.culture )
        #self.identity = self.assign_idenity()

        self.delta_t = delta_t

        #self.history_behaviour_list = [deepcopy(self.behaviour_list)]
        self.history_culture = [self.culture]

    def create_behaviours(self,init_data_behaviours):
        behaviour_list = []
        for i in range(len(init_data_behaviours)):
            behaviour_list.append(Behaviour(init_data_behaviours[i][0], init_data_behaviours[i][1], init_data_behaviours[i][2], init_data_behaviours[i][3], init_data_behaviours[i][4],init_data_behaviours[i][5]))
        return behaviour_list

    def calc_culture(self):
        total_culture = 0
        #print("out")
        for i in range(self.len_behaviour_list):
            if self.behaviour_list[i].behaviour_type == 1:
                total_culture += self.behaviour_list[i].value 
            else: 
                total_culture -= self.behaviour_list[i].value
            #print("inside culture count", total_culture)
        #the higher the cultural value the more it is pro envrionmental.

        av_culture = total_culture/self.len_behaviour_list
        #print("culture", av_culture)

        """
        if av_culture < -5:#cap the culture
            av_culture = -5
        elif av_culture > 5:
            av_culture = 5
        """

        return av_culture

    """
    def assign_idenity(self):
        if self.culture < 0:
            self.identity = "anti_environmental"
        elif 0 <= self.culture < 0.5:
            self.identity = "indifferent"
        elif self.culture >= 0:
            self.identity = "pro_environmental"
        else:
            raise ValueError('invalid cultural values')
    """
    
    def update_behaviours(self): # cultural_data, add in later the conformity bias
        for i in range(self.len_behaviour_list):
            self.behaviour_list[i].next_step()
            
    """
    def update_dynamics(self,neighbourhood_data,cultural_data):
        # step 2

    """
        
    def update_attracts(self,social_component_behaviours):
        #step 3, equation 2

        """
        #OPTION A: weighted difference
        for i in range(self.len_behaviour_list):
            #print(self.behaviour_list[i].attract, self.delta_t*(social_component_behaviours[i] - self.behaviour_list[i].attract), social_component_behaviours[i] - self.behaviour_list[i].attract, social_component_behaviours[i] )
            self.behaviour_list[i].attract += self.delta_t*(social_component_behaviours[i] - self.behaviour_list[i].attract) # add in + IP + FA later
        """ 

        """
        #OPTION B: weighted average, but its time independent
        for i in range(self.len_behaviour_list):
            self.behaviour_list[i].attract = (self.behaviour_list[i].attract + social_component_behaviours[i])/2 # add in + IP + FA later
            #print(self.behaviour_list[i].attract)
        """


        """
        #OPTION C: weighted average time
        for i in range(self.len_behaviour_list):
            self.behaviour_list[i].attract += self.delta_t*((self.behaviour_list[i].attract + social_component_behaviours[i])/2 ) # add in + IP + FA later
        """

        """
        for i in range(self.len_behaviour_list):
            #print("attract before",self.behaviour_list[i].attract)
            self.behaviour_list[i].attract += self.delta_t*(social_component_behaviours[i]) # add in + IP + FA later
            #print("attract after",self.behaviour_list[i].attract)
        """

        #OPTION D: RK2 weighted difference,  NEED TO CHECK, it looks wrong
        for i in range(self.len_behaviour_list):
            self.behaviour_list[i].attract += self.delta_t*(self.delta_t/2 + 1)*( self.behaviour_list[i].attract  - social_component_behaviours[i]) # add in + IP + FA later
        #print("attract after",self.behaviour_list[i].attract)

    def update_costs(self):
        # step 3, equation 
        extras = 0 #for now do nothing
        for i in range(self.len_behaviour_list):
            self.behaviour_list[i].attract += self.delta_t*(extras) # add in + CP and TC

    def update_culture(self):
        #print("before",self.culture )
        self.culture = self.calc_culture()
        #print("after",self.culture )

    def save_data_individual(self):
        #self.history_behaviour_list.append(deepcopy(self.behaviour_list))
        self.history_culture.append(self.culture)

    def next_step(self,social_component_behaviours):
        self.update_behaviours()#update the behaviours of agent
        #self.update_dynamics(neighbourhood_data,cultural_data) 
        self.update_attracts(social_component_behaviours)
        #self.update_costs()
        self.update_culture()
        self.save_data_individual()