import torch
import networkx as nx
import numpy as np
import os

import metrics as qm
import target_gen as ts

# sim_anneal uses the similarity function as described in the paper, sim_anneal_wasserstein uses the wasserstein distance and sinkhorn approximation
from sim_anneal import save_res, sim_anneal_torch
# from sim_anneal_wasserstein import save_res, sim_anneal_torch


"""
Main script to start morphing existing graph drawings into target shapes
 
Before you run this make sure you have existing edgelists and their coordinates in /data/edgelists/ and data/start_coords/, respectively

Change the 'graphs' variable to include more graphs
Change the 'metric_names' variable to include more metrics (change metric_names when you do this too)
Change the 'target_names' variable to include more targets

Change the import to sim_anneal_wasserstein if you want to use the Wasserstein Distance and Sinkhorn instead of the Similarity metric described in the paper
"""


if __name__ == '__main__':

    # enter the graphs that you would like to fool, their edgelists should be put in the data/edgeslists/ folder
    # their coordinates should be put in the data/start_coords/ folder
    graphs = ['bar_albert_gen', 'polbooks', 'gams10am', 'dwt_307', 'lnsp_131']

    # list the metrics and their names that you want to compute
    metrics = [qm.edge_lengths_sd_torch, qm.norm_stress_torch, qm.cross_pairs, qm.angular_resolution_dev]
    metric_names = ['ELD', 'ST', 'CN', 'AR']
    metric_dict = dict(zip(metric_names, metrics))

    # loop over all graphs
    for graph in graphs:

        print('Starting with: {}'.format(graph))

        # load the graph
        edl = np.loadtxt('data/edgelists/{}-edgelist.csv'.format(graph), delimiter = ',').astype(np.int64)
        G = nx.Graph()

        edgelist = []
        for e in edl:
            edgelist.append([e[0].item(), e[1].item()])
        G.add_edges_from(edgelist)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)

        gtds = nx.floyd_warshall_numpy(G)
        n = G.number_of_nodes()

        # load the start coordinates X and generate the target coordinates Y
        og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter = ',', dtype=np.float64)).float()
        np.random.seed(1337)
        targets_pos = [ts.circle_pos(n), ts.dinosaur(n), ts.lines(n = n, vert = True), ts.lines(n = n, vert = False), ts.cross(n), ts.grid(n)]
        targets_names = dict(zip(['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid'], targets_pos))

        # loop over all metrics
        for metric in metric_dict:

            print('Doing metric: {}'.format(metric))

            # get the initial quality metric value we want to get close to qm_0
            args_qm = [G, gtds, np.array(G.edges())]
            qm_target = metric_dict[metric](og_pos, args_qm)

            # setting epsilon, the margin that the metric can change
            if metric == 'CN':
                abs_diff = int(qm_target / 20)
            else:
                abs_diff = 0.0025

            # loop over all targets
            for target in targets_names:

                # don't have to recompute it if we already have results of it
                if not os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric)):

                    print('Replicating shape {}'.format(target))
                    tar_pos = torch.tensor(targets_names[target]).float()
                    args = [tar_pos, qm_target, metric_dict[metric]]

                    # main simulated annealing loop, adjust variables here if necessary (start_temp and max_N)
                    result = sim_anneal_torch(x0 = og_pos, args = args, args_qm = args_qm, start_temp = 0.4, max_N = 30000, abs_diff = abs_diff)

                    print('Similarity: {}'.format(str(round(result['sim'].item(), 4))))
                    print('Target QM val: {}'.format(str(round(result['qm_og'].item(), 4))))
                    print('Curr QM val: {}'.format(str(round(result['qm_new'].item(), 4))))
                    print('QM diff: {}'.format(str(round(np.abs(result['qm_diff'].item()), 4))))

                    title = 'Original {}: '.format(metric) + str(round(qm_target.item(), 4)) + ' | current {}: '.format(metric) + str(
                        round(result['qm_new'].item(), 4)) + ' | Similarity: ' + str(round(result['sim'].item(), 4))

                    save_res(result['coords'], G, graph_name = graph + '-' + target, metric_name = metric, title = title)
                else:
                    print('Already did this')
