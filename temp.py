import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

import target_gen as TG


def save_res(pos, G, full_name):

    pos_G = {k: pos[k] for k in range(len(pos))}
    nx.draw(G, with_labels=False, pos=pos_G, node_size=16, edge_color = ['lightblue'], width = 0.5, node_color = ['black'])
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.axis('off')
    plt.savefig('results/final_drawings/{}'.format(full_name), dpi = 400, bbox_inches = 'tight')
    plt.close('all')

n = 142
poss = [TG.circle_pos(n), TG.lines(n, True), TG.lines(n, False), TG.grid(n), TG.dinosaur(n), TG.cross(n)]
names = ['circle', 'vert-lines', 'hor-lines', 'grid', 'dinosaur', 'cross']
for i in range(len(poss)):
    og_pos = torch.tensor(poss[i])
    name = names[i]
    # scale translate pos
    pos = og_pos - torch.min(og_pos)
    pos /= torch.max(pos)
    full_name = 'target-{}.pdf'.format(name)

    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    save_res(pos, G, full_name = full_name)
