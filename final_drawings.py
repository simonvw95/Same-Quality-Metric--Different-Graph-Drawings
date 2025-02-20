import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


"""
Script to produce the final .pdf drawings of all your morphed drawings of your graphs in your dataset. Only works if you have the coordinates of your results 
after running main.py

Change the 'graphs' variable to include more graphs
Change the 'target_names' variable to include more targets
Change the 'metric_names' variable to include more metrics
"""


# there's multiple save_res functions but each is slightly modified, this one produces pdfs with larger node sizes to show the shapes, good for latex docs
def save_res(pos, G, full_name):

    pos_G = {k: pos[k] for k in range(len(pos))}
    nx.draw(G, with_labels = False, pos = pos_G, node_size = 16, edge_color = ['lightblue'], width = 0.5, node_color = ['black'])
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.axis('off')
    plt.savefig('results/final_drawings/{}'.format(full_name), dpi = 400, bbox_inches = 'tight')
    plt.close('all')


if __name__ == '__main__':

    # loop over all the graphs you have in your dataset
    graphs = ['bar_albert_gen', 'dwt_307', 'polbooks', 'gams10am', 'lnsp_131']
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

        # all target names and metrics
        targets_names = ['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid', 'init']
        metric_names = ['ELD', 'ST', 'CN', 'AR']

        for metric in metric_names:

            for target in targets_names:

                # might need to change paths here if your folder structure is different
                if os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric)):

                    # load the positioning, translate and scale them to [0,1]
                    og_pos = torch.tensor(np.loadtxt('results/{}-{}{}-coords.csv'.format(graph, target, metric), delimiter=',', dtype=np.float64)).float()
                    pos = og_pos - torch.min(og_pos)
                    pos /= torch.max(pos)

                    # save the pdf
                    full_name = graph + '-' + target + metric + '.pdf'
                    save_res(pos, G, full_name = full_name)

                # different path for the initial starting coordinates of the graphs
                if target == 'init':
                    og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
                    pos = og_pos - torch.min(og_pos)
                    pos /= torch.max(pos)

                    full_name = graph + '-init.pdf'
                    save_res(pos, G, full_name = full_name)
