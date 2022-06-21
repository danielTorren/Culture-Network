from behaviour import Behaviour
import numpy.typing as npt
class Individual:

    """
    Class for indivduals
    Properties: Culture, Behaviours list
    """

    def __init__(
        self, init_data_behaviours: list, delta_t: float, culture_momentum: int, t: float, M: int, save_data: bool, carbon_intensive_list: list
    ):
        self.M = M
        self.t = t
        self.delta_t = delta_t
        self.save_data = save_data
        self.carbon_intensive_list = carbon_intensive_list
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

    def create_behaviours(self, init_data_behaviours: list) -> list:
        # init_attract, init_threshold,carbon_emissions
        behaviour_list = [
            Behaviour(
                init_data_behaviours[i][0],
                init_data_behaviours[i][1],
                init_data_behaviours[i][2],
                self.delta_t,
                self.save_data,
            )
            for i in range(len(init_data_behaviours))
        ]
        return behaviour_list

    def update_av_behaviour_list(self):
        if len(self.av_behaviour_list) < self.culture_momentum:
            self.av_behaviour_list.append(self.av_behaviour)
        else:
            self.av_behaviour_list.pop(0)
            self.av_behaviour_list.append(self.av_behaviour)

    def calc_culture(self) -> float:
        self.update_av_behaviour_list()
        return sum(self.av_behaviour_list)/ self.culture_momentum


    def init_calc_carbon_emissions(self) -> float:
        total_emissions = 0
        for i in range(self.M):
            if self.behaviour_list[i].value <= 0:
                total_emissions += self.behaviour_list[i].carbon_emissions
        return total_emissions

    def init_calc_behaviour_av(self) -> float:
        total_behaviour = 0
        for i in range(self.M):
            total_behaviour += self.behaviour_list[i].value
        av_behaviour = total_behaviour / self.M

        return av_behaviour

    def behaviours_next_step(self, social_component_behaviours: npt.NDArray) -> tuple[float, float]:
        total_emissions = 0  # calc_carbon_emission
        total_behaviour = 0  # calc_behaviour_av
        for i in range(self.M):

            self.behaviour_list[i].next_step()  # update_behaviours
            self.behaviour_list[i].update_attract(
                social_component_behaviours[i]
            )  # update_attracts

            total_behaviour += self.behaviour_list[i].value  # calc_behaviour_av

            if (
                self.behaviour_list[i].value <= 0
            ):  # calc_carbon_emissions if less than or equal to 0 then it is a less environmetally friendly behaviour(brown)
                total_emissions += self.behaviour_list[
                    i
                ].carbon_emissions  # calc_carbon_emissions

        av_behaviour = total_behaviour / self.M  # calc_behaviour_av
        return total_emissions, av_behaviour  # calc_carbon_emissions #calc_behaviour_a

    def behaviours_next_step_alt(self, social_component_behaviours: npt.NDArray) -> tuple[float, float]:
        for i in range(self.M):

            self.behaviour_list[i].next_step()  # update_behaviours
            self.behaviour_list[i].update_attract(
                social_component_behaviours[i]
            )  # update_attracts

        return sum(self.carbon_intensive_list[i] for i in range(self.M) if self.behaviour_list[i].value <= 0), sum(self.behaviour_list[i].value for i in range(self.M))/self.M  # calc_carbon_emissions #calc_behaviour_av

    def save_data_individual(self):
        self.history_culture.append(self.culture)
        self.history_av_behaviour.append(self.av_behaviour)
        self.history_carbon_emissions.append(self.carbon_emissions)

    def next_step(self, social_component_behaviours: npt.NDArray):
        self.carbon_emissions, self.av_behaviour = self.behaviours_next_step_alt(
            social_component_behaviours
        )
        self.culture = self.calc_culture()

        if self.save_data:
            self.save_data_individual()
