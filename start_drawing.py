import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch

import criteria_torch_np as CNP

graph = 'lnsp_131'
edl = np.loadtxt('data/edgelists/{}.mtx'.format(graph), delimiter = ' ').astype(np.int64)[:, :2]
G = nx.Graph()
G.add_edges_from(edl)
G.remove_edges_from(nx.selfloop_edges(G))
G = nx.convert_node_labels_to_integers(G)
# for lnsp_131
# Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
# G = G.subgraph(Gcc[0])
# G = nx.convert_node_labels_to_integers(G)

edges = np.array(G.edges())
np.savetxt('data/edgelists/{}-edgelist.csv'.format(graph), edges, delimiter = ',')

pos_dict = nx.forceatlas2_layout(G, max_iter=5000)

pos = np.array([[float(pos_dict[k][0]), float(pos_dict[k][1])] for k in pos_dict])

# scale translate pos
pos = pos - np.min(pos)
pos /= np.max(pos)

qm_results = get_all_qms(pos, G)
stress = qm_results['ST']
eld = qm_results['ELD']
cross = qm_results['cross']
ar = qm_results['AR']

save_res(pos, G, graph_name=graph,
         title='Original drawing with ST: ' + str(stress) + ' | ELD: ' + str(eld) + ' | crossings: ' + str(
             cross) + ' | AR: ' + str(ar))


def get_all_qms(pos, G):
    gtds = nx.floyd_warshall_numpy(G)
    pos = torch.tensor(pos).float()

    stress = round(CNP.norm_stress_torch(pos, gtds).item(), 4)
    eld = round(CNP.edge_lengths_sd_torch(pos, np.array(list(G.edges()))).item(), 4)
    cross = round(CNP.cross_pairs(pos, np.array(list(G.edges()))).item(), 4)
    ar = round(CNP.angular_resolution_dev(np.array(pos), gtds), 4)

    return {'ST' : stress, 'ELD' : eld, 'cross' : cross, 'AR' : ar}


def save_res(pos, G, graph_name, title):

    plt.title(title, fontsize = 10)
    pos_G = {k: pos[k] for k in range(len(pos))}
    nx.draw(G, with_labels=False, pos=pos_G, node_size=16, edge_color = ['lightblue'], width = 1)
    plt.savefig('results/init-{}.png'.format(graph_name), dpi = 400)
    plt.close('all')

    # save the coordinates
    np.savetxt('data/start_coords/{}.csv'.format(graph_name), pos, delimiter = ',')

# edgelist from robot and ck104 wrong? get self loops
graphs = ['bar_albert_gen', 'robot', 'dwt_307', 'ck104']

for graph in graphs:
    print('doing {}'.format(graph))
    # load edgelist
    edl = np.loadtxt('data/edgelists/{}-edgelist.csv'.format(graph), delimiter = ',').astype(np.int64)

    # long step to keep node labelling consistent and gtds consistent
    temp_G = nx.Graph()
    edgelist = []
    for e in edl:
        edgelist.append([e[0].item(), e[1].item()])

    temp_G.add_edges_from(edgelist)
    G = nx.convert_node_labels_to_integers(temp_G)
    G.remove_edges_from(nx.selfloop_edges(G))

    pos_dict = nx.forceatlas2_layout(G, max_iter=5000)

    pos = np.array([[float(pos_dict[k][0]), float(pos_dict[k][1])] for k in pos_dict])

    # scale translate pos
    pos = pos - np.min(pos)
    pos /= np.max(pos)

    qm_results = get_all_qms(pos, G)
    stress = qm_results['ST']
    eld = qm_results['ELD']
    cross = qm_results['cross']
    ar = qm_results['AR']

    save_res(pos, G, graph_name = graph, title = 'Original drawing with ST: ' + str(stress) + ' | ELD: ' + str(eld) + ' | crossings: ' + str(cross) + ' | AR: ' + str(ar))
