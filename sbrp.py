import os

from rec_utils import NODE_INFECTION_PROBABILITY_ATTR, compute_neighbors_probability, remove_invalid_nodes, \
    init_extended_network, _get_nodes_to_process, remove_random_nodes

os.environ["PYTORCH_JIT"] = "0"
from inc.test import *
torch.set_printoptions(profile="full")

from inc.test import *

import torch
from networkx import Graph

def sbrp(
    G: Graph, IG: Graph, reconstruction_threshold=0.8, max_iterations: int = 5
) -> Graph:
    # flake8: noqa
    """SbRP graph reconstruction algorithm.

    @param G: Network
    @param IG: Infected network
    @param reconstruction_threshold: Reconstruction threshold

    @return: Extended network
    References
        ----------
        - [1] W. Zang, P. Zhang, C. Zhou, i L. Guo, „Discovering Multiple
        Diffusion Source Nodes in Social Networks”, Procedia Comput. Sci.,
        t. 29, s. 443–452, 2014, doi: 10.1016/j.procs.2014.05.040.
        - [2] W. Zang, P. Zhang, C. Zhou, i L. Guo, „Locating multiple sources
        in social networks under the SIR model: A divide-and-conquer approach”,
         J. Comput. Sci., t. 10, s. 278–287, wrz. 2015,
         doi: 10.1016/j.jocs.2015.05.002.
    """
    EG = init_extended_network(G=G, IG=IG)
    iter = 0
    nodes = _get_nodes_to_process(EG=EG, IG=IG, threshold=reconstruction_threshold)

    while iter < max_iterations and nodes:
        iter += 1
        for node in IG:
            for neighbour in G.neighbors(node):
                if neighbour in IG:
                    continue
                propability = compute_neighbors_probability(G=EG, node=neighbour)
                EG.nodes[neighbour][NODE_INFECTION_PROBABILITY_ATTR] = propability

        nodes = _get_nodes_to_process(EG=EG, IG=IG, threshold=reconstruction_threshold)

    result = remove_invalid_nodes(EG,IG, reconstruction_threshold)
    return result



@torch.no_grad()
def sbrp_run(data, **kwargs):
    T = int(data.T.item())
    n_nodes = int(data.num_nodes)
    n_cls = int(data.y.max().item()) + 1

    G = pyg.utils.to_networkx(data, to_undirected=True, remove_self_loops=True)

    node_ids = sorted(G.nodes())
    id2i = {nid: i for i, nid in enumerate(node_ids)}

    y = data.y.detach().cpu().numpy()  # (n_nodes, T+1)
    y_pred = np.zeros((n_nodes, T + 1), dtype=np.int32)
    remove_ratio = 0.1

    for t in range(T + 1):
        # obserwacja na czas t (1 = I, 0 = S/R)
        infected_idx = np.where((y[:, t] & 1) == 1)[0]
        infected_idx = remove_random_nodes(infected_idx, remove_ratio)
        IG_t = G.subgraph(infected_idx).copy()

        EG_t = sbrp(G=G, IG=IG_t)

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
    tester = Tester(cfg['data_dir'], cfg['device'], sbrp_run)
    tester.test([cfg['dataset']], rep=1)
    tester.save(cfg['output'])

def main():
    args = get_args()
    run(vars(args))

if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)
    tester = Tester(args.data_dir, args.device, sbrp_run)
    tester.test([args.dataset], rep = 1)
    tester.save(args.output)