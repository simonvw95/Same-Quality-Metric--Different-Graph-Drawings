import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch

import metrics as qm


"""
Script to produce the initial coordinates of your graphs in your dataset using FA2. Also produces .png images of these initial coordinates including
information on the quality metric values of the initial drawings

Change the 'graphs' variable to include more graphs
Change the 'get_all_qms' function to include more quality metrics
"""


# function that acquires all the quality metrics concerned in the paper, extend this if you want to add more qms (pay attention to if they use np or torch)
def get_all_qms(pos, args_qm):

    pos = torch.tensor(pos).float()

    stress = round(qm.norm_stress_torch(pos, args_qm).item(), 4)
    eld = round(qm.edge_lengths_sd_torch(pos, args_qm).item(), 4)
    cross = round(qm.cross_pairs(pos, args_qm).item(), 4)
    ar = round(qm.angular_resolution_dev(np.array(pos), args_qm), 4)

    return {'ST' : stress, 'ELD' : eld, 'CN' : cross, 'AR' : ar}


# function to save the graph drawings as png with a title including the metric values, and saves the produced coordinates
def save_res(pos, G, graph_name, title):

    plt.title(title, fontsize = 10)
    pos_G = {k: pos[k] for k in range(len(pos))}
    nx.draw(G, with_labels = False, pos = pos_G, node_size = 16, edge_color = ['lightblue'], width = 1)
    plt.savefig('results/init-{}.png'.format(graph_name), dpi = 400)
    plt.close('all')

    # save the coordinates
    np.savetxt('data/start_coords/{}.csv'.format(graph_name), pos, delimiter = ',')


if __name__ == '__main__':

    # loop over all the graphs in your dataset
    graphs = ['bar_albert_gen', 'polbooks', 'gams10am', 'dwt_307', 'lnsp_131']

    for graph in graphs:

        print('doing {}'.format(graph))

        # load edgelist
        edl = np.loadtxt('data/edgelists/{}-edgelist.csv'.format(graph), delimiter = ',').astype(np.int64)

        temp_G = nx.Graph()
        edgelist = []
        for e in edl:
            edgelist.append([e[0].item(), e[1].item()])

        temp_G.add_edges_from(edgelist)
        G = nx.convert_node_labels_to_integers(temp_G)
        G.remove_edges_from(nx.selfloop_edges(G))

        pos_dict = nx.forceatlas2_layout(G, max_iter = 5000)

        pos = np.array([[float(pos_dict[k][0]), float(pos_dict[k][1])] for k in pos_dict])

        # scale translate pos
        pos = pos - np.min(pos)
        pos /= np.max(pos)

        args_qm = [G, nx.floyd_warshall_numpy(G), np.array(G.edges())]
        qm_results = get_all_qms(pos, args_qm)
        stress = qm_results['ST']
        eld = qm_results['ELD']
        cross = qm_results['CN']
        ar = qm_results['AR']

        save_res(pos, G, graph_name = graph, title = 'Original drawing with ST: ' + str(stress) + ' | ELD: ' + str(eld) + ' | CN: ' + str(cross) + ' | AR: ' + str(ar))
