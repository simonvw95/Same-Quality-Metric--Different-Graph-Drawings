import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
from PIL import Image

import metrics as qmf


"""
Script to produce the final .pdf drawings of all your morphed drawings of your graphs in your dataset. Only works if you have the coordinates of your results
after running main.py

Change the 'graphs' variable to include more graphs
Change the 'target_names' variable to include more targets
Change the 'metric_names' variable to include more metrics
"""


# format arrays: floats get 4 decimals (fixed width), ints stay ints
def format_array(arr, decimals=4, width=8):
    def fmt(x):
        if float(x).is_integer():  # keep integers clean
            return f"{int(x):{width}d}"
        else:  # format floats with fixed width & 4 decimals
            return f"{x:{width}.{decimals}f}"
    return np.array2string(arr, formatter={'float_kind': fmt, 'int_kind': fmt}, separator=' ')


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

        og_pos = torch.tensor(np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
        args = G, nx.floyd_warshall_numpy(G), np.array(G.edges())

        # all target names and metrics
        metric_names = ['ST', 'AR', 'CN', 'ELD']
        full_metric_names = dict(zip(metric_names, ['Stress', 'Angular Resolution', 'Number of Crossings', 'Edge Length Deviation']))
        metric_funcs = dict(zip(metric_names, [qmf.norm_stress_torch, qmf.angular_resolution_dev, qmf.cross_pairs, qmf.edge_lengths_sd_torch]))
        targets_names = []

        # CHANGE HERE
        # metric combinations
        metric_combs = [['ST', 'ELD', 'CN', 'AR'], 'ST', 'AR', 'CN', 'ELD']

        for file in sorted(os.listdir('rickroll/frame_coords/'), key=extract_number):
            targets_names.append(file)
        targets_names.append('init')

        for metric in metric_combs:

            all_png_names = []

            if isinstance(metric, list):
                qm_init = {}
                for sub_m in metric:
                    qm_init[sub_m] = metric_funcs[sub_m](og_pos, args).item()
                qm_init_list = torch.tensor(list(qm_init.values()))
            else:
                qm_init_list = metric_funcs[metric](og_pos, args)

            for target in targets_names:

                # load the positioning, translate and scale them to [0,1]
                file_metric = metric
                if isinstance(metric, list):
                    file_metric = str(metric).replace("'", "").replace(", ", "-").replace("[", "").replace("]", "")

                # might need to change paths here if your folder structure is different
                if os.path.exists('rickroll/{}-{}{}-coords.csv'.format(graph, target, file_metric)):

                    pos = torch.tensor(np.loadtxt('rickroll/{}-{}{}-coords.csv'.format(graph, target, file_metric), delimiter=',', dtype=np.float64)).float()
                    # pos = og_pos - torch.min(og_pos)
                    # pos /= torch.max(pos)
                    if isinstance(metric, list):
                        qm_val = {}
                        for sub_m in metric:
                            qm_val[sub_m] = metric_funcs[sub_m](pos, args).item()
                        qm_new = torch.tensor(list(qm_val.values()))
                    else:
                        qm_new = metric_funcs[metric](pos, args)

                    if isinstance(metric, list):
                        title = ('Original {}:\n'.format(metric) + format_array(qm_init_list.numpy()) +
                                 '\nCurrent  {}:\n'.format(metric) + format_array(qm_new.numpy()))
                        fontsize_title = 20
                        x_cutoff = -0.025
                    else:
                        title = ('Original {}: '.format(metric) + format_array(qm_init_list.numpy()) +
                                 '\nCurrent  {}: '.format(metric) + format_array(qm_new.numpy()))
                        fontsize_title = 25
                        x_cutoff = -0.075
                    # create figure/axes explicitly (important for stability)
                    fig, ax = plt.subplots(figsize=(6, 6))

                    # set the title
                    ax.set_title(title, fontsize=fontsize_title, rotation='vertical', x=x_cutoff, y=0.1)
                    # pos = (pos - torch.min(pos)) / (torch.max(pos) - torch.min(pos))
                    # convert nodes to dictionary for networkx
                    pos_G = {k: list(pos[k]) for k in G.nodes()}
                    pos_G = {k: [pos_G[k][0].item(), pos_G[k][1].item()] for k in pos_G}

                    # fix axes, prevents the drawing from sometimes bouncing around due to matplotlib stuff
                    ax.set_xlim(-0.25, 1.25)
                    ax.set_ylim(-0.25, 1.25)
                    ax.set_aspect('equal')  # lock aspect ratio
                    ax.autoscale(False)  # prevent rescaling
                    ax.set_position([0.15, 0.01, 0.99, 0.99])

                    # draw network
                    nx.draw(G, with_labels=False, pos=pos_G, node_size=5, edge_color=['lightblue'], width=0.5,
                            node_color=['black'], ax=ax)

                    # save figure
                    full_name = '{}-{}{}.png'.format(graph, target, file_metric)
                    plt.savefig('rickroll/' + full_name)
                    plt.close(fig)

                    all_png_names.append(full_name)

            # make gif

            # get all the PNG files in the directory
            images = [Image.open(os.path.join('rickroll/', filename)) for filename in all_png_names if filename.endswith('.png')]

            # rotate all images
            for i in range(len(images)):
                images[i] = images[i].rotate(270, Image.NEAREST, expand = 1)

            # add the reversed version of the list, remove the first and last element
            rev = images[::-1]
            rev.pop(0)
            rev.pop(-1)

            images += rev

            # define the output gif file path
            gif_output_path = 'rickroll/rr-{}-{}.gif'.format(graph, metric)

            # save the images as a gif at 24 frames per second
            images[0].save(gif_output_path, save_all = True, append_images = images[1:], optimize = False, duration = 1000//10, loop=0)

            print(f"GIF saved as {gif_output_path}")

        # save the initial drawing
        full_name_init = graph + '-init.png'
        metric = ['ST', 'ELD', 'CN', 'AR']
        qm_init = {}
        for sub_m in metric:
            qm_init[sub_m] = metric_funcs[sub_m](og_pos, args).item()
        qm_init_list = torch.tensor(list(qm_init.values()))

        title = ('Original {}: '.format(metric) + format_array(qm_init_list.numpy()))

        # create figure/axes explicitly (important for stability)
        fig, ax = plt.subplots(figsize=(6, 6))

        # set the title
        ax.set_title(title, fontsize=10)

        # convert nodes to dictionary for networkx
        pos_G = {k: list(og_pos[k]) for k in G.nodes()}
        pos_G = {k: [pos_G[k][0].item(), pos_G[k][1].item()] for k in pos_G}

        # fix axes, prevents the drawing from sometimes bouncing around due to matplotlib stuff
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')  # lock aspect ratio
        ax.autoscale(False)  # prevent rescaling
        ax.set_position([0.15, 0.01, 0.9, 0.8])

        # draw network
        nx.draw(G, with_labels=False, pos=pos_G, node_size=16, edge_color=['lightblue'], width=1,
                node_color=['black'], ax=ax)

        # save figure
        plt.savefig('rickroll/' + full_name_init)
        plt.close(fig)
