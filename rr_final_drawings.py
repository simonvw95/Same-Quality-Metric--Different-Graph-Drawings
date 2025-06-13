import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
from PIL import Image

import metrics as qm

"""
Script to produce the final .pdf drawings of all your morphed drawings of your graphs in your dataset. Only works if you have the coordinates of your results 
after running main.py

Change the 'graphs' variable to include more graphs
Change the 'target_names' variable to include more targets
Change the 'metric_names' variable to include more metrics
"""


# there's multiple save_res functions but each is slightly modified, this one produces pdfs with larger node sizes to show the shapes, good for latex docs
def save_res(pos, G, full_name, text):

    pos_G = {k: pos[k] for k in range(len(pos))}
    fig, axes = plt.subplots(1, 2, gridspec_kw = {'width_ratios': [4, 1]})
    ax = axes.flatten()
    for i in range(2):
        if i == 0:
            nx.draw(G, with_labels = False, pos = pos_G, node_size = 16, edge_color = ['lightblue'], width = 0.5, node_color = ['black'], ax = ax[i])
            # ax[i].set_xlim(-0.01, 1.01)
            # ax[i].set_ylim(-0.01, 1.01)
            ax[i].set_axis_off()
        if i == 1:
            ax[i].set_axis_off()
            ax[i].text(0.5, 0.5, text, rotation = 90)
    # nx.draw(G, with_labels = False, pos = pos_G, node_size = 16, edge_color = ['lightblue'], width = 0.5, node_color = ['black'])
    # plt.axis('off')
    plt.savefig('results/final_drawings/{}'.format(full_name), dpi = 400, bbox_inches = 'tight')
    plt.close('all')

    plt.plot()


# Function to extract the numeric part from the filename for sorting
def extract_number(filename):
    match = re.search(r'(\d+)', filename)  # Find the first number in the filename
    return int(match.group(1)) if match else 0


if __name__ == '__main__':

    # loop over all the graphs you have in your dataset
    graphs = ['ba_rr']
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
        metric_names = ['ELD', 'ST', 'CN', 'AR']
        full_metric_names = dict(zip(metric_names, ['Edge Length Deviation', 'Stress', 'Number of Crossings', 'Angular Resolution']))
        metric_funcs = dict(zip(metric_names, [qm.edge_lengths_sd_torch, qm.norm_stress_torch, qm.cross_pairs, qm.angular_resolution_dev]))
        targets_names = []
        qms_init = []

        for file in sorted(os.listdir('rickroll/frame_coords/'), key=extract_number):
            targets_names.append(file)
        targets_names.append('init')

        for metric in metric_names:

            all_png_names = []
            args = G, nx.floyd_warshall_numpy(G), np.array(G.edges())

            for target in targets_names:

                # might need to change paths here if your folder structure is different
                if os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric)):

                    # load the positioning, translate and scale them to [0,1]
                    pos = torch.tensor(np.loadtxt('results/{}-{}{}-coords.csv'.format(graph, target, metric), delimiter=',', dtype=np.float64)).float()
                    # pos = og_pos - torch.min(og_pos)
                    # pos /= torch.max(pos)
                    qm_val = metric_funcs[metric](pos, args)

                    # save the pdf
                    full_name = graph + '-' + target + metric + '.png'
                    all_png_names.append(full_name)
                    text = full_metric_names[metric] + ': \n' + str(round(qm_val.item(), 5))
                    save_res(pos, G, full_name = full_name, text = text)

            # different path for the initial starting coordinates of the graphs
            if target == 'init':
                og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
                pos = og_pos - torch.min(og_pos)
                pos /= torch.max(pos)

                qm_val = metric_funcs[metric](pos, args)
                qms_init.append(qm_val)

            # make gif

            # get all the PNG files in the directory
            images = [Image.open(os.path.join('results/final_drawings/', filename)) for filename in all_png_names if filename.endswith('.png')]

            # rotate all images
            for i in range(len(images)):
                images[i] = images[i].rotate(270, Image.NEAREST, expand = 1)

            # add the reversed version of the list, remove the first and last element
            rev = images[::-1]
            rev.pop(0)
            rev.pop(-1)

            images += rev

            # define the output gif file path
            gif_output_path = 'results/rr-{}-{}.gif'.format(graph, metric)

            # save the images as a gif at 24 frames per second
            images[0].save(gif_output_path, save_all = True, append_images = images[1:], optimize = False, duration = 1000//10, loop=0)

            print(f"GIF saved as {gif_output_path}")

        # save the initial drawing
        full_name = graph + '-init.png'
        text = ''
        for i in range(len(metric_names)):
            text += '\n' + full_metric_names[metric_names[i]] + ': \n' + str(round(qms_init[i].item(), 4)) + '\n'

        save_res(pos, G, full_name = full_name, text = text)
