import networkx as nx
import matplotlib.pyplot as plt
import torch

import target_gen as tg


"""
Script to produce .pdf images of all the target coordinates (not graph drawings) of your target shapes, currently set number of points is 142

Change the 'poss' variable to include more functions capable of creating target coordinates
Change the 'names' variable to indicate the names of these new functions
Change the 'n' variable to lower or increase the number of points in the target shape
"""


def save_res(pos, G, full_name):

    pos_G = {k: pos[k] for k in range(len(pos))}
    nx.draw(G, with_labels = False, pos = pos_G, node_size = 16, edge_color = ['lightblue'], width = 0.5, node_color = ['black'])
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.axis('off')
    plt.savefig('results/final_drawings/{}'.format(full_name), dpi = 400, bbox_inches = 'tight')
    plt.close('all')


if __name__ == '__main__':

    n = 142
    poss = [tg.circle_pos(n), tg.lines(n, True), tg.lines(n, False), tg.grid(n), tg.dinosaur(n), tg.cross(n)]
    names = ['circle', 'vert-lines', 'hor-lines', 'grid', 'dinosaur', 'cross']

    for i in range(len(poss)):

        # load the positioning, translate and scale them to [0,1]
        og_pos = torch.tensor(poss[i])
        pos = og_pos - torch.min(og_pos)
        pos /= torch.max(pos)

        name = names[i]
        full_name = 'target-{}.pdf'.format(name)

        # create a graph without edges just so we can use networkx to save JUST node coordinates
        G = nx.Graph()
        G.add_nodes_from(list(range(n)))

        save_res(pos, G, full_name = full_name)
