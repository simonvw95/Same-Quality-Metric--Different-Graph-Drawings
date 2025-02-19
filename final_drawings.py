import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def save_res(pos, G, full_name):

    pos_G = {k: pos[k] for k in range(len(pos))}
    nx.draw(G, with_labels=False, pos=pos_G, node_size=16, edge_color = ['lightblue'], width = 0.5, node_color = ['black'])
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.axis('off')
    plt.savefig('results/final_drawings/{}'.format(full_name), dpi = 400, bbox_inches = 'tight')
    plt.close('all')


graphs = ['gams10am']  # ['bar_albert_gen', 'dwt_307', 'polbooks', 'gams10am', 'lnsp_131']
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

    targets_names = ['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'init', 'grid']
    metric_names = ['ELD', 'ST', 'CROSS', 'AR']

    for metric in metric_names:

        for target in targets_names:

            # change paths here to final drawings
            if os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric)):
                og_pos = torch.tensor(np.loadtxt('results/{}-{}{}-coords.csv'.format(graph, target, metric), delimiter=',', dtype=np.float64)).float()
                # scale translate pos
                pos = og_pos - torch.min(og_pos)
                pos /= torch.max(pos)
                full_name = graph + '-' + target + metric + '.pdf'

                save_res(pos, G, full_name = full_name)

            if target == 'init':
                og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
                # scale translate pos
                pos = og_pos - torch.min(og_pos)
                pos /= torch.max(pos)
                full_name = graph + '-init.pdf'

                save_res(pos, G, full_name = full_name)
