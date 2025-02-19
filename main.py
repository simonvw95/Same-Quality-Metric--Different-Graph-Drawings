import torch
import networkx as nx
import numpy as np
import os

import criteria_torch_np as CNP
import target_gen as TG
from sim_anneal import save_res, sim_anneal_torch

graphs = ['dwt_307']# 'gams10am']  # ,'bar_albert_gen', 'polbooks', 'gams10am', 'dwt_307', 'dwt_307'] #['bar_albert_gen']# 'dwt_307']

metrics = [CNP.edge_lengths_sd_torch, CNP.norm_stress_torch, CNP.cross_pairs, CNP.angular_resolution_dev]
metric_names = dict(zip(metrics, ['ELD', 'ST', 'CROSS', 'AR']))
# temp changes, only do grid with old parameters for all graphs
for graph in graphs:

    print('Starting with: {}'.format(graph))
    # load the graph
    edl = np.loadtxt('data/edgelists/{}-edgelist.csv'.format(graph), delimiter=',').astype(np.int64)
    G = nx.Graph()
    # something went wrong with either the edge list or the coordinates but this fixes it
    if graph == 'gams10am' or graph == 'polbooks' or graph == 'lnsp_131':
        G.add_nodes_from(list(range(0, np.max(edl) + 1)))

    edgelist = []
    for e in edl:
        edgelist.append([e[0].item(), e[1].item()])
    G.add_edges_from(edgelist)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    gtds = nx.floyd_warshall_numpy(G)
    n = G.number_of_nodes()

    og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
    np.random.seed(1337)
    targets_pos = [TG.circle_pos(n), TG.dinosaur(n), TG.lines(n = n, vert = True), TG.lines(n = n, vert = False), TG.cross(n), TG.grid(n)]
    targets_names = dict(zip(['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid'], targets_pos))

    for metric in metrics:

        print('Doing metric: {}'.format(metric.__name__))

        abs_diff = 0.0025
        if metric.__name__ == 'norm_stress_torch':
            qm_target = metric(og_pos, gtds)
        elif metric.__name__ == 'edge_lengths_sd_torch':
            qm_target = metric(og_pos, np.array(G.edges()))
        elif metric.__name__ == 'angular_resolution_dev':
            qm_target = torch.tensor(metric(np.array(og_pos), gtds))
        elif metric.__name__ == 'cross_pairs':
            qm_target = metric(og_pos, np.array(G.edges()))
            abs_diff = int(qm_target / 20)

        for target in targets_names:

            curr_metric_name = metric_names[metric]
            if not os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, curr_metric_name)):

                print('Replicating shape {}'.format(target))
                tar_pos = torch.tensor(targets_names[target]).float()
                args = [tar_pos, qm_target, metric, G]
                result = sim_anneal_torch(x0 = og_pos, args = args, start_temp = 0.4, max_N = 30000, abs_diff = abs_diff)

                print('Wasserstein Distance: ' + str(round(result['WD'].item(), 4)))
                print('Target QM val: ' + str(round(result['qm_og'].item(), 4)))
                print('Curr QM val: ' + str(round(result['qm_new'].item(), 4)))
                print('QM diff: ' + str(round(np.abs(result['qm_diff'].item()), 4)))

                title = 'Original {}: '.format(curr_metric_name) + str(round(qm_target.item(), 4)) + ' | current {}: '.format(curr_metric_name) + str(
                    round(result['qm_new'].item(), 4)) + ' | Wasserstein: ' + str(round(result['WD'].item(), 4))

                save_res(result['coords'], G, graph_name = graph + '-' + target, metric_name = curr_metric_name, title = title)
            else:
                print('Already did this')
