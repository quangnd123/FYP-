from Trajectory_Network import TrajectoryNetwork
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random


class HeterogeneousSEIR():
    def __init__(self, trajectory_network: TrajectoryNetwork, s_initial: set, i_initial: set, infect_rate: float, t_incubation: float, t_recovery: float, time_step: float):
        self.trajectory_network = trajectory_network
        # SEIR at each time step
        self.SEIR_population = [[s_initial, set(), i_initial, set()]]

        self.infect_rate = infect_rate

        # timestep is the duration for a trajectory graph
        self.t_incubation = math.ceil(t_incubation/time_step)
        self.t_recovery = math.ceil(t_recovery/time_step)

        # incubation period of each person
        self.incubation_periods = defaultdict(int)
        # recovery period of each person
        self.recovery_periods = defaultdict(int)

        self.ids = s_initial.union(i_initial)

        self.epidemic_spreading()

    def calculate_infect_probability(self, contact_duration: float) -> float:
        """The probability of infection between 2 individuals after contact_duration, given their total_contact_duration_in_the_same_timestep """
        # see Stehlé et al. BMC Medicine 2011
        infect_probability = self.infect_rate*contact_duration
        return infect_probability

    def epidemic_spreading(self):
        # see the algorithm in PechlivanogLou et al. 2022
        graphs = self.trajectory_network.get_trajectory_network()

        for index, graph in enumerate(graphs):
            S = self.SEIR_population[index][0]
            E = self.SEIR_population[index][1]
            I = self.SEIR_population[index][2]
            R = self.SEIR_population[index][3]

            next_S = S.copy()
            next_E = E.copy()
            next_I = I.copy()
            next_R = R.copy()

            for u in self.ids:
                if u in S:  # u is suspected
                    if not u in graph:
                        continue
                    for v in graph.neighbors(u):
                        if v in I:  # v is infected
                            # v does not infect u
                            if random.random() > self.calculate_infect_probability(graph[u][v]["weight"]):
                                continue

                            # incubation period of u begins
                            self.incubation_periods[u] = 0

                            if u in next_S:
                                next_S.remove(u)
                            next_E.add(u)
                elif u in E:  # u is exposed
                    self.incubation_periods[u] += 1
                    if self.incubation_periods[u] == self.t_incubation:
                        # recovery period of u begins
                        self.recovery_periods[u] = 0
                        next_E.remove(u)
                        next_I.add(u)
                elif u in I:  # u is infected
                    self.recovery_periods[u] += 1
                    if self.recovery_periods[u] == self.t_recovery:
                        next_I.remove(u)
                        next_R.add(u)
                else:
                    pass

            self.SEIR_population.append([next_S, next_E, next_I, next_R])
        return

    def get_num_SEIR(self):
        # dim: days x 4
        # the number people of each group S, E, I , R over time
        return [[len(s) for s in SEIR_t] for SEIR_t in self.SEIR_population]
