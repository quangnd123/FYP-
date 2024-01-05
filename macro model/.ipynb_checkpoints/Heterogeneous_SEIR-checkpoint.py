from Trajectory_Network import TrajectoryNetwork
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random


class HeterogeneousSEIR():
    def __init__(self, trajectory_network: TrajectoryNetwork, s_initial: set, i_initial: set, infect_rate: float, exposed2infectious_prob: float, recover_prob: float, q: float, t_incubation: float, t_infectious: float, effective_duration: float, time_step: float):
        self.trajectory_network = trajectory_network
        # SEIR at each time step
        self.SEIR = [[s_initial, set(), i_initial, set()]]

        self.infect_rate = infect_rate
        self.exposed2infectious_prob = exposed2infectious_prob
        self.recover_prob = recover_prob

        # timestep is the duration for a trajectory graph
        self.t_incubation = math.ceil(t_incubation/time_step)
        self.t_infectious = math.ceil(t_infectious/time_step)

        # quarantine rate
        self.q = q
        # Q[i] = 1 if i is in quarantine, 0 otherwise
        self.Qs = defaultdict(int)

        # incubation period of each person
        self.incubation_periods = defaultdict(int)
        # recovery period of each person
        self.recovery_periods = defaultdict(int)

        # the totaltime during which the links in the daily networks wereconsidered active
        # assume timestep is oneday, effective duration is daytime
        self.effective_duration = effective_duration

        self.nodes = s_initial.union(i_initial)

        self.epidemic_spreading()

    def calculate_infect_probability(self, contact_duration: float, total_contact_duration_in_a_day: float) -> float:
        """The probability of infection between 2 individuals after contact_duration, given their total_contact_duration_in_the_same_timestep """
        # see Stehlé et al. BMC Medicine 2011
        infect_probability = self.infect_rate*contact_duration
        return infect_probability
#         # Why rescale? Because heterogeneous netowrk does not preserve causality constraint.
#         rescaled_infect_probability = infect_probability * \
#             total_contact_duration_in_a_day/self.effective_duration
#         return min(1, rescaled_infect_probability)

    def epidemic_spreading(self):
        # see the algorithm in PechlivanogLou et al. 2022
        graphs = self.trajectory_network.get_trajectory_network()
        
        for index, graph in enumerate(graphs):
            S = self.SEIR[index][0]
            E = self.SEIR[index][1]
            I = self.SEIR[index][2]
            R = self.SEIR[index][3]

            next_S = S.copy()
            next_E = E.copy()
            next_I = I.copy()
            next_R = R.copy()

            for u in self.nodes:
                if u in S:  # u is suspected
                    if not u in graph:
                        continue
                    for v in graph.neighbors(u):
                        if v in I and self.Qs.get(v, 0) != 1:  # v is infected
                            # v does not infect u
                            if random.random() > self.calculate_infect_probability(graph[u][v]["weight"], graph[u][v]["weight"]):
                                continue

                            self.Qs[u] = 1 if random.random() <= self.q else 0

                            # incubation period of u begins
                            self.incubation_periods[u] = 0

                            if u in next_S:
                                next_S.remove(u)
                            next_E.add(u)
                elif u in E:  # u is exposed
                    self.incubation_periods[u] += 1
                    if self.incubation_periods[u] == self.t_incubation:
                        if random.random() <= self.exposed2infectious_prob:  # u moves to Infected group
                            # recovery period of u begins
                            self.recovery_periods[u] = 0
                            next_E.remove(u)
                            next_I.add(u)
                        else:  # u moves back to Susceptible group
                            next_E.remove(u)
                            next_S.add(u)

                elif u in I:  # u is infected
                    self.recovery_periods[u] += 1
                    if self.recovery_periods[u] == self.t_infectious:
                        next_I.remove(u)

                        if random.random() <= self.recover_prob:  # if u recovers
                            next_R.add(u)
                        else:  # u dies
                            pass
                else:
                    pass

            self.SEIR.append([next_S, next_E, next_I, next_R])
        return
    
    def get_no_SEIR(self):
        return [[len(s) for s in SEIR_t] for SEIR_t in self.SEIR] 
    