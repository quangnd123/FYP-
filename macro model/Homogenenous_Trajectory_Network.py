from Trajectory_Network import TrajectoryNetwork
from typing import List
from tqdm import tqdm
import networkx as nx
import pandas as pd


class HomogeneousTrajectoryNetwork(TrajectoryNetwork):
    def __init__(self, list_of_dfs: List[pd.DataFrame]):
        super().__init__(list_of_dfs)
        self.build_network()

    def build_network(self):
        self.graphs = [self.build_graph(df) for df in tqdm(self.list_of_dfs, desc="Building Trajectory Network")]
        return

    def build_graph(self, df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, row in df.iterrows():
            if not G.has_node(row['id_1']):
                G.add_node(row['id_1'])
            if not G.has_node(row['id_2']):
                G.add_node(row['id_2'])
            if row['id_1'] != row['id_2']:
                if G.has_edge(row['id_1'], row['id_2']):
                    G[row['id_1']][row['id_2']]['weight'] += row['contact_duration']
                else:
                    G.add_edge(row['id_1'], row['id_2'],
                               weight=row['contact_duration'])
                    
        # Calculate the mean of existing edge weights
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]            
        mean_weight = sum(edge_weights) / len(edge_weights) if len(edge_weights) > 0 else 0
        
        # The link between 2 individuals in the graph is the mean contact duration in the whole df
        # see Stehlé et al. BMC Medicine 2011

        # Set all edge weights to the mean
        for u, v, data in G.edges(data=True):
            G[u][v]['weight'] = mean_weight
        return G
