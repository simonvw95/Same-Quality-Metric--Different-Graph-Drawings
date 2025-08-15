import torch
import networkx as nx
import numpy as np
import os
import re

import metrics as qm
from sim_anneal_mult import save_res, sim_anneal_mult



"""
Main script to start morphing existing graph drawings into target shapes

Before you run this make sure you have existing edgelists and their coordinates in /data/edgelists/ and data/start_coords/, respectively

Change the 'graphs' variable to include more graphs
Change the 'metric_names' variable to include more metrics (change metric_names when you do this too)
Change the 'target_names' variable to include more targets

Change the import to sim_anneal_wasserstein if you want to use the Wasserstein Distance and Sinkhorn instead of the Similarity metric described in the paper
"""


# Function to extract the numeric part from the filename for sorting
def extract_number(filename):
    match = re.search(r'(\d+)', filename)  # Find the first number in the filename
    return int(match.group(1)) if match else 0


if __name__ == '__main__':

    # enter the graphs that you would like to fool, their edgelists should be put in the data/edgeslists/ folder
    # their coordinates should be put in the data/start_coords/ folder
    graphs = ['ba_rr']

    # list the metrics and their names that you want to compute
    metrics = [qm.edge_lengths_sd_torch, qm.norm_stress_torch, qm.cross_pairs, qm.angular_resolution_dev]
    metric_names = ['ELD', 'ST', 'CN', 'AR']
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

        targets_pos = []
        names = []

        for file in sorted(os.listdir('rickroll/frame_coords/'), key=extract_number):

            targets_pos.append(np.loadtxt('rickroll/frame_coords/{}'.format(file), delimiter=',', dtype=np.float64))
            names.append(file)

        targets_names = dict(zip(names, targets_pos))

        # CHANGE QUALITY METRIC COMBINATIONS HERE
        metric_combs = ['ST', 'ELD', 'CN', 'AR']
        metric_name = 'ST-ELD-CN-AR'
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
            if not os.path.exists('rickroll/{}-{}{}-coords.csv'.format(graph, target, metric_name)):

                print('Replicating shape {}'.format(target))
                tar_pos = torch.tensor(targets_names[target]).float()
                args = [tar_pos, qm_targets, qm_funcs]

                # main simulated annealing loop, adjust variables here if necessary (start_temp and max_N)
                result = sim_anneal_mult(x0=og_pos, args=args, args_qm=args_qm, start_temp=0.4, max_N=300, abs_diffs=abs_diffs)

                print('Similarity: {}'.format(str(round(result['sim'].item(), 4))))
                print('Target QM val: {}'.format(str(result['qm_og'].numpy().round(4))))
                print('Curr QM val: {}'.format(str(result['qm_new'].numpy().round(4))))
                print('QM diff: {}'.format(str(result['qm_diff'].numpy().round(4))))

                title = 'Original {}: '.format(metric_name) + str(qm_targets.numpy().round(4)) + '\ncurrent {}: '.format(
                    metric_name) + str(result['qm_new'].numpy().round(4)) + '\nSimilarity: ' + str(result['sim'].numpy().round(4))

                save_res(result['coords'], G, graph_name=graph + '-' + target, metric_name=metric_name, title=title, rr=True)
            else:
                print('Already did this')
