import os
from dataclasses import dataclass
from typing import Set

from networkx.algorithms.centrality import degree_centrality

from rec_utils import init_extended_network, _get_nodes_to_process, compute_neighbors_probability, \
    _check_if_node_is_on_path_between_infected_nodes, NODE_INFECTION_PROBABILITY_ATTR, remove_invalid_nodes, \
    remove_random_nodes

os.environ["PYTORCH_JIT"] = "0"
from inc.test import *
torch.set_printoptions(profile="full")

from inc.test import *

from functools import cached_property

import torch
from networkx import Graph

@dataclass
class PropagationReconstructionConfig:
    G: Graph
    IG: Graph
    real_IG: Graph

    m0: float = -0.85
    m1: float = 61.74
    m2: float = -47.80
    max_iterations: int = 5
    threshold: float = 0.8

    @cached_property
    def observed_infected_nodes(self) -> Set:
        return set(self.IG)

    @cached_property
    def real_infected_nodes(self) -> Set:
        return set(self.real_IG)

def shni(
    config: PropagationReconstructionConfig
) -> Graph:
    # flake8: noqa
    """SHNI graph reconstruction algorithm."""

    G=config.G
    IG = config.IG

    EG = init_extended_network(G=G, IG=IG)
    iter = 0
    nodes = [1]
    degree_c = degree_centrality(EG)
    while iter < config.max_iterations and nodes:
        iter += 1
        for node in IG:
            for neighbour in G.neighbors(node):
                if neighbour in IG:
                    continue
                degree = degree_c[neighbour]
                neighbors_probability = compute_neighbors_probability(node, EG)
                node_on_path = _check_if_node_is_on_path_between_infected_nodes(node=node, G=EG, threshold=config.threshold)

                m0 = config.m0
                m1 = config.m1 * neighbors_probability * degree
                m2 = config.m2 * node_on_path * degree

                z = m0 + m1 + m2
                z = max(min(z, 500), -500)
                probability = 1 / (1 + math.exp(-z))

                EG.nodes[neighbour][NODE_INFECTION_PROBABILITY_ATTR] = probability

        nodes = _get_nodes_to_process(EG=EG,IG=IG, threshold=config.threshold)

    result = remove_invalid_nodes(EG,IG, config.threshold)
    return result



@torch.no_grad()
def shni_run(data, **kwargs):
    T = int(data.T.item())
    n_nodes = int(data.num_nodes)
    n_cls = int(data.y.max().item()) + 1

    G = pyg.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)

    node_ids = sorted(G.nodes())
    id2i = {nid: i for i, nid in enumerate(node_ids)}

    y = data.y.detach().cpu().numpy()  # (n_nodes, T+1)
    y_pred = np.zeros((n_nodes, T + 1), dtype=np.int32)

    for t in range(T + 1):
        # obserwacja na czas t (1 = I, 0 = S/R)
        infected_idx = np.where((y[:, t] & 1) == 1)[0]
        infected_idx = remove_random_nodes(infected_idx, 0.1)

        IG_t = G.subgraph(infected_idx).copy()

        cfg = PropagationReconstructionConfig(
            G=G, IG=IG_t, real_IG=IG_t
        )
        EG_t = shni(cfg)

        col = np.zeros(n_nodes, dtype=np.int32)

        for nid in EG_t.nodes():
            col[id2i[nid]] = 1

        y_pred[:, t] = col

    y_pred = np.minimum(y_pred, n_cls - 1)
    return torch.tensor(y_pred, dtype=torch.long, device=data.y.device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int, default=1, help='number of tests per dataset')
    parser.add_argument('--dataset', type=str, default='farmers-si', help='dataset name')
    parser.add_argument('--seed', type=int, default=123456789, help='random seed')
    parser.add_argument('--data_dir', type=str, default='input', help='dataset folder')
    parser.add_argument('--output', type=str, default='output/dhrec-farmers-si.pt', help='output file name')

    args = parser.parse_args()
    return args

def run(cfg):
    seed_all(cfg['seed'])
    tester = Tester(cfg['data_dir'], cfg['device'], shni_run)
    tester.test([cfg['dataset']], rep=1)
    tester.save(cfg['output'])

def main():
    args = get_args()
    run(vars(args))

if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)
    tester = Tester(args.data_dir, args.device, shni_run)
    tester.test([args.dataset], rep = 1)
    tester.save(args.output)