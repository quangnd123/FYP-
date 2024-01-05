from collections import defaultdict
import random
import pandas as pd
import numpy as np


class DynamicSEIR():
    def __init__(self, df: pd.DataFrame, s_initial: set, i_initial: set, infect_rate: float, t_incubation: float, t_recovery: float, t_loss_immunity: float):
        self.df = df

        self.susceptible_population = s_initial.copy()
        self.exposed_population = set()
        self.infectious_population = i_initial.copy()
        self.recovered_population = set()

        # SEIR at each time step
        self.SEIR_population = {0: [self.susceptible_population.copy(), self.exposed_population.copy(
        ), self.infectious_population.copy(), self.recovered_population.copy()]}

        # infect probability depends on infect rate and contact duration
        self.infect_rate = infect_rate

        # Exposed -> Infectious after t_incubation
        self.t_incubation = t_incubation
        # Infectious -> Recovered after t_recovery
        self.t_recovery = t_recovery
        # Recovered -> Suceptible after t_loss_immunity
        self.t_loss_immunity = t_loss_immunity

        # incubation starting time point of each person
        self.incubation_start_moment = defaultdict(int)
        # recovery starting time point of each person
        self.recovery_start_moment = defaultdict(int)
        # losing immunity starting time point of each person
        self.loss_immunity_start_moment = defaultdict(int)

        # all people's id
        self.ids = s_initial.union(i_initial)

        self.epidemic_spreading()

    def calculate_infect_probability(self, contact_duration: float) -> float:
        """The probability of infection between 2 individuals after contact_duration, given their total_contact_duration_in_the_same_timestep """
        # see Stehlé et al. BMC Medicine 2011
        infect_probability = self.infect_rate*contact_duration
        return infect_probability

    def exposed_to_infectious_moment(self, id) -> float:
        return self.t_incubation + self.incubation_start_moment[id]

    def infectious_to_recovered_moment(self, id) -> float:
        return self.t_recovery + self.recovery_start_moment[id]

    def recovered_to_susceptible_moment(self, id) -> float:
        return self.t_loss_immunity + self.loss_immunity_start_moment[id]

    def susceptible_to_exposed(self, id, moment):
        self.incubation_start_moment[id] = moment
        self.susceptible_population.discard(id)
        self.exposed_population.add(id)

    def exposed_to_infectious(self, id, moment):
        self.recovery_start_moment[id] = moment
        self.exposed_population.discard(id)
        self.infectious_population.add(id)

    def infectious_to_recovered(self, id, moment):
        self.loss_immunity_start_moment[id] = moment
        self.infectious_population.discard(id)
        self.recovered_population.add(id)

    def recovered_to_susceptible(self, id):
        self.recovered_population.discard(id)
        self.susceptible_population.add(id)

    def Susceptible_and_Exposed_interaction(self, contact_end_moment, interval_end_moment, id_1, id_2):
        # id_1: Susceptible and id_2: Exposed

        t = self.exposed_to_infectious_moment(id_2)
        if t <= interval_end_moment:
            self.exposed_to_infectious(id=id_2, moment=t)
            if t <= contact_end_moment:
                # the last contact moment of id_1 and id_2 during the current interval
                last_contact_moment_during_interval = min(
                    interval_end_moment, contact_end_moment)
                if random.random() <= self.calculate_infect_probability(last_contact_moment_during_interval-t):
                    self.susceptible_to_exposed(
                        id=id_1, moment=last_contact_moment_during_interval)

    def Susceptible_and_Infectious_interaction(self, contact_start_moment, contact_end_moment, interval_end_moment, id_1, id_2):
        # id_1: Susceptible and id_2: Infectious

        t = self.infectious_to_recovered_moment(id=id_2)
        if t <= interval_end_moment:
            self.infectious_to_recovered(id=id_2, moment=t)
            # the last contact moment of id_1 and id_2 during the current interval before id_2 becomes recovered
            last_contact_moment_during_interval_when_id_2_infectious = min(
                t, contact_end_moment)
            if random.random() <= last_contact_moment_during_interval_when_id_2_infectious:
                self.susceptible_to_exposed(
                    id=id_1, moment=last_contact_moment_during_interval_when_id_2_infectious)
        else:
            # the last contact moment of id_1 and id_2 during the current interval
            last_contact_moment_during_interval = min(
                interval_end_moment, contact_end_moment)
            if random.random() <= self.calculate_infect_probability(last_contact_moment_during_interval-contact_start_moment):
                self.susceptible_to_exposed(
                    id=id_1, moment=last_contact_moment_during_interval)

    def Exposed_and_Recovered_interation(self, contact_end_moment, interval_end_moment, id_1, id_2):
        # id_1: Exposed and id_2: Recovered
        t1 = self.exposed_to_infectious_moment(id=id_1)
        t2 = self.recovered_to_susceptible_moment(id=id_2)

        if t1 <= interval_end_moment:
            self.exposed_to_infectious(id=id_1, moment=t1)
        if t2 <= interval_end_moment:
            self.recovered_to_susceptible(id=id_2)

        if max(t1, t2) <= min(interval_end_moment, contact_end_moment):
            if random.random() <= self.calculate_infect_probability(min(interval_end_moment, contact_end_moment)):
                self.susceptible_to_exposed(id=id_2, moment=min(
                    interval_end_moment, contact_end_moment))

    def Infectious_and_Recovered_interaction(self, contact_end_moment, interval_end_moment, id_1, id_2):
        # id_1: Infectious and id_2: Recovered
        t1 = self.infectious_to_recovered_moment(id=id_1)
        t2 = self.recovered_to_susceptible_moment(id=id_2)

        if t1 <= interval_end_moment:
            self.infectious_to_recovered(id=id_1, moment=t1)
        if t2 <= interval_end_moment:
            self.recovered_to_susceptible(id=id_2)

        if t2 <= t1 and t2 <= contact_end_moment and t2 <= interval_end_moment:
            if random.random() <= self.calculate_infect_probability(min(t1, contact_end_moment, interval_end_moment) - t2):
                self.susceptible_to_exposed(id=id_2, moment=min(
                    t1, contact_end_moment, interval_end_moment))

    def epidemic_spreading(self):
        df = self.df
        moments = np.sort(np.unique(df[['start_moment', 'end_moment']]))

        start_row_index = 0

        for idx, interval_start_moment in enumerate(moments):
            if idx == len(moments)-1:
                break

            interval_end_moment = moments[idx+1]
            processed_ids = set()

            susceptible_population = self.susceptible_population.copy()
            exposed_population = self.exposed_population.copy()
            infectious_population = self.infectious_population.copy()
            recovered_population = self.recovered_population.copy()

            # process all contacts during the interval
            for _, row in df.iloc[start_row_index:].iterrows():
                if row["end_moment"] <= interval_start_moment:
                    start_row_index += 1
                    continue
                elif interval_start_moment < row["start_moment"]:
                    break

                # when row["start_moment"] <= interval_start_moment < row["end_moment"]

                if row["id_1"] in susceptible_population and row["id_2"] in exposed_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Susceptible_and_Exposed_interaction(
                        contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_1"], id_2=row["id_2"])
                elif row["id_2"] in susceptible_population and row["id_1"] in exposed_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Susceptible_and_Exposed_interaction(
                        contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_2"], id_2=row["id_1"])
                elif row["id_1"] in susceptible_population and row["id_2"] in infectious_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Susceptible_and_Infectious_interaction(
                        contact_start_moment=row["start_moment"], contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_1"], id_2=row["id_2"])
                elif row["id_2"] in susceptible_population and row["id_1"] in infectious_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Susceptible_and_Infectious_interaction(
                        contact_start_moment=row["start_moment"], contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_2"], id_2=row["id_1"])
                elif row["id_1"] in exposed_population and row["id_2"] in recovered_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Exposed_and_Recovered_interation(
                        contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_1"], id_2=row["id_2"])
                elif row["id_2"] in exposed_population and row["id_1"] in recovered_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Exposed_and_Recovered_interation(
                        contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_2"], id_2=row["id_1"])
                elif row["id_1"] in infectious_population and row["id_2"] in recovered_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Infectious_and_Recovered_interaction(
                        contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_1"], id_2=row["id_2"])
                elif row["id_2"] in infectious_population and row["id_1"] in recovered_population:
                    processed_ids.add(row["id_1"])
                    processed_ids.add(row["id_2"])
                    self.Infectious_and_Recovered_interaction(
                        contact_end_moment=row["end_moment"], interval_end_moment=interval_end_moment, id_1=row["id_2"], id_2=row["id_1"])
                else:
                    pass

            # update other people's status
            for id in self.ids:
                if id not in processed_ids:
                    if id in exposed_population:
                        t = self.exposed_to_infectious_moment(id)
                        if t <= interval_end_moment:
                            self.exposed_to_infectious(id=id, moment=t)
                    elif id in infectious_population:
                        t = self.infectious_to_recovered_moment(id)
                        if t <= interval_end_moment:
                            self.infectious_to_recovered(id=id, moment=t)
                    elif id in recovered_population:
                        t = self.recovered_to_susceptible_moment(id)
                        if t <= interval_end_moment:
                            self.recovered_to_susceptible(id=id)
                    else:
                        pass

            self.SEIR_population[interval_end_moment] = [self.susceptible_population.copy(
            ), self.exposed_population.copy(), self.infectious_population.copy(), self.recovered_population.copy()]

        return

    def get_num_SEIR(self):
        # dim: days x 4
        # the number people of each group S, E, I , R over time
        return [[len(s) for s in SEIR_t] for end_time, SEIR_t in self.SEIR_population.items()]
