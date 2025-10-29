import itertools
from functools import lru_cache
from typing import List

import networkx as nx
import numpy as np
from networkx.classes import Graph

NODE_INFECTION_PROBABILITY_ATTR = "INFECTION_PROBABILITY"


@lru_cache
def get_shortest_path(G: Graph, source: int, target: int) -> List[int]:
    return nx.shortest_path(G, source=source, target=target)


def _get_nodes_to_process(EG: Graph, IG: Graph, threshold: float) -> List[int]:
    ig_nodes = set(IG.nodes())

    neighbor_candidates = set()
    for v in ig_nodes:
        if v in EG:
            neighbor_candidates.update(EG.neighbors(v))
    neighbor_candidates -= ig_nodes

    nodes = [
        n for n in neighbor_candidates
        if NODE_INFECTION_PROBABILITY_ATTR in EG.nodes[n]
           and EG.nodes[n][NODE_INFECTION_PROBABILITY_ATTR] < threshold
    ]

    def risky_neighbor_count(n):
        return sum(
            1
            for nn in EG.neighbors(n)
            if NODE_INFECTION_PROBABILITY_ATTR in EG.nodes[nn]
            and EG.nodes[nn][NODE_INFECTION_PROBABILITY_ATTR] > threshold
        )

    return sorted(nodes, key=risky_neighbor_count, reverse=True)


def _check_if_node_is_on_path_between_infected_nodes(node: int,
                                                     G: Graph, threshold: float) -> float:
    # neighbors = list(nx.neighbors(G, node))
    # combination = list(itertools.combinations(neighbors, 2))
    # sum = 0.0
    # for n1, n2 in combination:
    #     sp = get_shortest_path(G, source=n1, target=n2)
    #     if node in sp:
    #         sum += 1.0
    # value = sum / len(combination) if sum > 0 else 0.0
    # return value

    nbrs = [v for v in G.neighbors(node)
            if G[node][v].get(NODE_INFECTION_PROBABILITY_ATTR, 0.0) >= threshold]
    k = len(nbrs)
    if k < 2:
        return 0.0

    total = k * (k - 1) // 2
    # policz krawędzie (n1, n2) wśród sąsiadów, które także spełniają próg
    m = 0
    nbrs_set = set(nbrs)
    for n1, n2 in itertools.combinations(nbrs, 2):
        if G.has_edge(n1, n2) and G[n1][n2].get(NODE_INFECTION_PROBABILITY_ATTR, 0.0) >= threshold:
            m += 1

    # pary niepołączone / wszystkie pary
    return (total - m) / total



def init_extended_network(G: Graph, IG: Graph) -> Graph:
    """
    Initialize extended network.

    @param G: Network
    @param IG: Infected network

    @return: Extended network
    """
    EG = G.copy()
    nx.set_node_attributes(
        EG,
        {
            node: {NODE_INFECTION_PROBABILITY_ATTR: 1.0 if node in IG else 0.0}
            for node in G
        },
    )
    return EG


def compute_neighbors_probability(node: int, G: Graph) -> float:
    """
    Compute probability of infection for a given node.

    @param node: Node
    @param G: Graph

    @return: Probability of infection for a given node
    """
    neighbors_probability = [
        G.nodes[node][NODE_INFECTION_PROBABILITY_ATTR] for node in nx.neighbors(G, node)
    ]
    return sum(neighbors_probability) * 1.0 / len(neighbors_probability)


def remove_invalid_nodes(EG: Graph,IG: Graph, threshold: float) -> Graph:
    """
    Remove nodes with infection probability lower than threshold.

    @param EG: Extended network
    @param threshold: Infection probability threshold

    @return: Extended network with removed nodes that have infection probability lower than threshold
    """
    nodes_to_remove = []
    for node in EG.nodes(data=True):
        data = node[1]
        infection_probability = data[NODE_INFECTION_PROBABILITY_ATTR]
        if infection_probability < threshold:
            nodes_to_remove.append(node[0])

    EG.remove_nodes_from(nodes_to_remove)
    EG.remove_nodes_from(list(nx.isolates(EG)))
    EG.add_nodes_from(IG.nodes())
    return EG

def remove_random_nodes(infected_idx, remove_ratio=0.1):
    if remove_ratio > 0 and len(infected_idx) > 0:
        drop_n = max(1, int(0.1 * len(infected_idx)))
        drop_idx = np.random.choice(infected_idx, size=drop_n, replace=False)
        infected_idx = np.setdiff1d(infected_idx, drop_idx)
    return infected_idx

