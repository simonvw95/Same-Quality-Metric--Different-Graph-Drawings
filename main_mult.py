import torch
import networkx as nx
import numpy as np
import os

import metrics as qmf
import target_gen as ts

# sim_anneal uses the similarity function as described in the paper, sim_anneal_wasserstein uses the wasserstein distance and sinkhorn approximation
from sim_anneal_mult import save_res, sim_anneal_mult

np.set_printoptions(suppress=True)


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

    # list the metrics and their names that you want to compute, select the combinations down below
    metrics = [qmf.norm_stress_torch, qmf.angular_resolution_dev, qmf.cross_pairs, qmf.edge_lengths_sd_torch]
    metric_names = ['ST', 'AR', 'CN', 'ELD']
    metric_dict = dict(zip(metric_names, metrics))

    # loop over all graphs
    for graph in graphs:

        print('Starting with: {}'.format(graph))

        # load the graph
        edl = np.loadtxt('data/edgelists/{}-edgelist.csv'.format(graph), delimiter=',').astype(np.int64)
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
        og_pos = torch.tensor(
            np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
        np.random.seed(1337)
        targets_pos = [ts.circle_pos(n), ts.dinosaur(n), ts.lines(n=n, vert=True), ts.lines(n=n, vert=False),
                       ts.cross(n), ts.grid(n)]
        targets_names = dict(zip(['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid'], targets_pos))

        # CHANGE COMBINATION OF METRICS HERE
        metric_combs = ['ST', 'AR', 'CN', 'ELD']
        metric_name = 'ST-AR-CN-ELD'
        # get the initial quality metric value we want to get close to qm_0
        args_qm = [G, gtds, np.array(G.edges())]

        qm_targets = []
        abs_diffs = []
        qm_funcs = []

        for qm in metric_combs:
            qm_funcs.append(metric_dict[qm])
            curr_qm_target = metric_dict[qm](og_pos, args_qm)
            qm_targets.append(curr_qm_target.item())

            # setting epsilon, the margin that the metric can change
            if qm == 'CN':
                abs_diff = int(curr_qm_target / 20)
            else:
                abs_diff = 0.0025

            abs_diffs.append(abs_diff)

        qm_targets = torch.tensor(qm_targets)
        abs_diffs = torch.tensor(abs_diffs)

        # loop over all targets
        for target in targets_names:

            # don't have to recompute it if we already have results of it
            if not os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric_name)):

                print('Replicating shape {}'.format(target))
                tar_pos = torch.tensor(targets_names[target]).float()
                args = [tar_pos, qm_targets, qm_funcs, metric_dict]

                # set name_parts to None in case you don't want to iteratively save the coordinates of the process, if set to None only the end product is saved of the alg
                # name_parts = None
                name_parts = ['results', graph, target, metric_combs]

                # main simulated annealing loop, adjust variables here if necessary (start_temp and max_N)
                result = sim_anneal_mult(x0=og_pos, args=args, args_qm=args_qm, start_temp=0.4, max_N=30000, abs_diffs=abs_diffs, name_parts=name_parts)

                print('Similarity: {}'.format(str(round(result['sim'].item(), 4))))
                print('Target QM val: {}'.format(str(result['qm_og'].numpy().round(4))))
                print('Curr QM val: {}'.format(str(result['qm_new'].numpy().round(4))))
                print('QM diff: {}'.format(str(result['qm_diff'].numpy().round(4))))

                title = 'Original {}: '.format(metric_name) + str(qm_targets.numpy().round(4)) + '\ncurrent {}: '.format(
                    metric_name) + str(result['qm_new'].numpy().round(4)) + '\nSimilarity: ' + str(result['sim'].numpy().round(4))

                save_res(result['coords'], G, graph_name=graph + '-' + target, metric_name=metric_name, title=title, rr=False)
            else:
                print('Already did this')
