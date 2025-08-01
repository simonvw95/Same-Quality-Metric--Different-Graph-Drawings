import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from matplotlib import rc

# change to latex font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


import target_gen as ts
from metrics import similarity

"""
Script to produce the final .pdfs of the similarity percentages all your morphed drawings of your graphs in your dataset. Only works if you have the coordinates of your results 
after running main.py

Change the 'graphs' variable to include more graphs
Change the 'target_names' variable to include more targets
Change the 'metric_names' variable to include more metrics
"""


# there's multiple save_res functions but each is slightly modified, this one produces pdfs with larger node sizes to show the shapes, good for latex docs
def save_res(og_pos, target_pos, start_pos, full_name, value = None):

    if value == 0:
        sim_value = 0
    else:
        sim_start_target = similarity(start_pos, torch.tensor(target_pos))
        sim_og_target = similarity(og_pos, torch.tensor(target_pos))
        sim_value = 100 - ((sim_og_target.item() / sim_start_target.item()) * 100)

    plt.text(s = "{:.2f}\%".format(sim_value), x = 0.5, y = 0.5, fontsize = 110, ha = 'center', va = 'center')
    plt.axis('off')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.savefig('results/final_simvals/{}'.format(full_name), dpi = 400, bbox_inches = 'tight')
    plt.close('all')


if __name__ == '__main__':

    # loop over all the graphs you have in your dataset
    graphs = ['bar_albert_gen', 'dwt_307', 'polbooks', 'gams10am', 'lnsp_131']

    all_results = {}
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

        # COMBS METRICS
        metric_names = ['ST-AR-CN-ELD', 'ST-AR-ELD', 'ST-AR-CN', 'ST-CN-ELD', 'AR-CN-ELD', 'AR-CN', 'AR-ELD', 'CN-ELD', 'ST-AR', 'ST-CN', 'ST-ELD', 'ELD', 'ST', 'CN', 'AR']

        n = G.number_of_nodes()

        # load the start coordinates X and generate the target coordinates Y
        start_pos = torch.tensor(
            np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
        np.random.seed(1337)
        targets_pos = [ts.circle_pos(n), ts.dinosaur(n), ts.lines(n=n, vert=True), ts.lines(n=n, vert=False),
                       ts.cross(n), ts.grid(n), start_pos]
        targets_names_dict = dict(zip(['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid', 'init'], targets_pos))

        metric_results = {}
        for metric in metric_names:

            target_results = {}
            for target in targets_names:

                target_pos = targets_names_dict[target]
                # might need to change paths here if your folder structure is different
                if os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric)):

                    # load the positioning, translate and scale them to [0,1]
                    og_pos = torch.tensor(np.loadtxt('results/{}-{}{}-coords.csv'.format(graph, target, metric), delimiter=',', dtype=np.float64)).float()
                    # pos = og_pos - torch.min(og_pos)
                    # pos /= torch.max(pos)

                    # save the pdf
                    full_name = graph + '-' + target + metric + '.pdf'
                    save_res(og_pos, target_pos, start_pos, full_name = full_name)

                # different path for the initial starting coordinates of the graphs
                if target == 'init':
                    og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
                    # pos = og_pos - torch.min(og_pos)
                    # pos /= torch.max(pos)

                    full_name = graph + '-init.pdf'
                    save_res(og_pos, target_pos, start_pos, full_name = full_name, value = 0)
