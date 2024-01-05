from collections import defaultdict
from typing import List
import networkx as nx
import pandas as pd


class TrajectoryNetwork():
    def __init__(self, list_of_dfs: List[pd.DataFrame]):
        # df must have these columns: "id_1", "id_2", "contact_duration"
        # "contact_duration" is the duration of a contact between "id_1" and "id_2"
        self.list_of_dfs = list_of_dfs

        # a sequence of graphs, each graph represents contacts in a timestep
        self.graphs = []

        self.risk_1s = {}  # risk1 of individuals
        self.risk_2s = {}  # risk2 of individuals
        self.risk_3s = {}  # risk3 of individuals
        self.sum_risk_1s = 0
        self.sum_risk_2s = 0
        self.sum_risk_3s = 0

        # contact duration of i and j
        self.contact_durations = defaultdict(lambda: defaultdict(int))

        # self.calculate_contact_durations()
        # self.calculate_risk_1()
        # self.calculate_risk_2()
        # self.calculate_risk_3()

    def calculate_contact_durations(self):
        """contact durations for all pairs of individuals"""
        for df in self.list_of_dfs:
            for idx, row in df.iterrows():
                if row["id_1"] != row["id_2"]:
                    self.contact_durations[row["id_1"]
                                           ][row["id_2"]] += row["contact_duration"]
                    self.contact_durations[row["id_2"]
                                           ][row["id_1"]] += row["contact_duration"]
        return

    def calculate_risk_1(self):
        """The total number of contact for each individual in the temporal network"""
        # see Eq.(1) in PechlivanogLou et al. 2022
        for id, inner_dict in self.contact_durations.items():
            self.risk_1s[id] = len(inner_dict)
            self.sum_risk_1s += self.risk_1s[id]
        return

    def calculate_risk_2(self):
        """The total duration of contacts for each individual in the temporal network"""
        # see Eq.(2) in PechlivanogLou et al. 2022
        for id, inner_dict in self.contact_durations.items():
            self.risk_2s[id] = sum(inner_dict.values())
            self.sum_risk_2s += self.risk_2s[id]
        return

    # def calculate_risk_3(self):
    #     """The probability of getting infected by any of its contacts factoring the total duration of these contacts"""
    #     # see Eq.(3) in PechlivanogLou et al. 2022
    #     for id, inner_dict in self.contact_durations.items():
    #         # Is it the correct way to get beta
    #         self.risk_3s[id] = sum((1-(1-self.calculate_infect_probability(duration, duration))**duration) for duration in inner_dict.values())
    #         self.sum_risk_3s += self.risk_3s[id]

    def get_risk_1(self, id) -> float:
        return self.risk_1s.get(id, 0)

    def get_risk_2(self, id) -> float:
        return self.risk_2s.get(id, 0)

    def get_risk_3(self, id) -> float:
        return self.risk_3s.get(id, 0)

    def get_relative_risk_1(self, id) -> float:
        return 0 if self.sum_risk_1s == 0 else self.risk_1s.get(id, 0)/self.sum_risk_1s

    def get_relative_risk_2(self, id) -> float:
        return 0 if self.sum_risk_2s == 0 else self.risk_2s.get(id, 0)/self.sum_risk_2s

    def get_relative_risk_3(self, id) -> float:
        return 0 if self.sum_risk_3s == 0 else self.risk_3s.get(id, 0)/self.sum_risk_3s

    def get_trajectory_network(self) -> List[nx.Graph]:
        return self.graphs
