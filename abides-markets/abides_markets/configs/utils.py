from abides_core.network import BANetwork, Network
from abides_core import CommunicativeAgent
from typing import List, Generator
import numpy as np
from copy import deepcopy


def generate_BA_graph_sequence(starting_graph: Network, m: int, new_agents: List[CommunicativeAgent],
                               graph_sizes: List[int], random_state: np.random.RandomState = None) -> Generator[
    Network, None, None]:
    for graph_size in iter(graph_sizes):
        agents_in_graph = starting_graph.size
        agents_to_add = graph_size - agents_in_graph
        new_agents_batch = new_agents[:agents_to_add]
        new_agents = new_agents[agents_to_add:]
        new_graph = BANetwork.construct_from_agent_list(starting_graph, new_agents_batch, m, random_state)
        yield new_graph
        starting_graph = new_graph

