import numpy as np
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import torch
import warnings
import re
import imageio
import metrics as qmf
import target_gen as ts
from tqdm import tqdm


# function to extract the numeric part from the filename for sorting
def extract_number(filename):
    match = re.search(r'(\d+)', filename)  # Find the first number in the filename
    return int(match.group(1)) if match else 0


# format arrays: floats get 4 decimals (fixed width), ints stay ints
def format_array(arr, decimals=4, width=8):
    def fmt(x):
        if float(x).is_integer():  # keep integers clean
            return f"{int(x):{width}d}"
        else:  # format floats with fixed width & 4 decimals
            return f"{x:{width}.{decimals}f}"
    return np.array2string(arr, formatter={'float_kind': fmt, 'int_kind': fmt}, separator=' ')


# ignoring warnings and better printing in matplotlib
np.seterr(divide = 'ignore', invalid = 'ignore')
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
plt.rcParams['font.family'] = 'monospace'

spec_dir = 'results/'

# specify the graph and metric combination you want to process the json
# this script was not intended to process ALL jsons of ALL graphs and metric combs automatically since I did not want to generate all the jsons
# necessary for those, only a handful are needed to get the point across
graph = 'bar_albert_gen'  # change this if necessary
metric_combs = ['ST', 'AR', 'CN', 'ELD']  # change this if necessary
metric_name = 'ST-AR-CN-ELD'  # change this if necessary

metric_abbr = ['ST', 'AR', 'CN', 'ELD']
metrics_f = [qmf.norm_stress_torch, qmf.angular_resolution_dev, qmf.cross_pairs, qmf.edge_lengths_sd_torch]
metric_dict = dict(zip(metric_abbr, metrics_f))

# load the graph data
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

# load the start coordinates X
og_pos = np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)
args_qm = [G, gtds, np.array(G.edges())]

qm_targets = []

for qm in metric_combs:
    curr_qm_target = metric_dict[qm](torch.tensor(og_pos), args_qm)
    qm_targets.append(curr_qm_target.item())
qm_targets = torch.tensor(qm_targets)

np.random.seed(1337)
targets_pos = [ts.circle_pos(n), ts.dinosaur(n), ts.lines(n=n, vert=True), ts.lines(n=n, vert=False),
               ts.cross(n), ts.grid(n), ts.rand_pos(n)]
targets_names = dict(zip(['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid', 'random'], targets_pos))


# now that the data is loaded we can process the json files to gifs
# targets = ['circle', 'cross', 'dinosaur', 'grid', 'hor-lines', 'vert-lines']
targets = ['random']
for target in targets:

    # target position
    tar_pos = targets_names[target]
    # starting dissimilarity
    start_similarity = qmf.similarity(torch.tensor(og_pos), torch.tensor(tar_pos))

    file = spec_dir + '{}-{}{}.json'.format(graph, target, str(metric_combs))

    with open(file, 'r') as f:
        data = json.load(f)

    # create a temporary folder for storing pngs to process into a gif
    if not os.path.exists(spec_dir + 'tempstorage/'):
        os.mkdir(spec_dir + 'tempstorage/')

    # loop over all the iterations in the json file
    for i in tqdm(data):
        # new coordinates
        curr_pos = np.array(data[i]['coords'])
        # new quality metrics of the new coordinates
        qm_new = []
        for q in metric_combs:
            qm_new.append(data[i]['qms'][q])
        qm_new = torch.tensor(qm_new)

        # similarity to starting coordinates in raw value and percentage value
        similarity = qmf.similarity(torch.tensor(curr_pos), torch.tensor(tar_pos))
        sim_perc = 100 - (similarity.item() / start_similarity.item() * 100)

        title = ('Original {}: '.format(metric_name) + format_array(qm_targets.numpy()) +
                 '\nCurrent  {}: '.format(metric_name) + format_array(qm_new.numpy()) +
                 '\nSimilarity: Raw Value: {:<8.4f} | Perc Value: {:<6.2f}%'.format(similarity.item(), sim_perc))

        # create figure/axes explicitly (important for stability)
        fig, ax = plt.subplots(figsize=(6, 6))

        # shrink margins to use most of the figure
        # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.0, top=0.85)  # leave space for title

        # set the title
        ax.set_title(title, fontsize=10)

        # convert nodes to dictionary for networkx
        # pos = (curr_pos - np.min(curr_pos)) / (np.max(curr_pos) - np.min(curr_pos))
        pos_G = {k: list(curr_pos[k]) for k in G.nodes()}
        pos_G = {k: [pos_G[k][0].item(), pos_G[k][1].item()] for k in pos_G}

        # fix axes, prevents the drawing from sometimes bouncing around due to matplotlib stuff
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')  # lock aspect ratio
        ax.autoscale(False)  # prevent rescaling
        ax.set_position([0.05, 0.01, 0.9, 0.8])

        # draw network
        nx.draw(G, with_labels=False, pos=pos_G, node_size=16, edge_color=['lightblue'], width=1, node_color=['black'], ax=ax)

        # save figure
        plt.savefig(spec_dir + 'tempstorage/' + '{}-{}{}-{}.png'.format(graph, target, str(metric_combs), i))
        plt.close(fig)

    save_dir = spec_dir + 'final_drawings/'
    # get all the PNG files in the directory (sorted by the numeric part of the filename and if it has the target)
    images = [os.path.join(spec_dir + 'tempstorage/', filename) for filename in
              sorted(os.listdir(spec_dir + 'tempstorage/'), key=extract_number) if (filename.endswith('.png') and filename.rfind(target) != -1)]

    if len(images) == 0:
        raise Exception('No images found for the target')

    # load the frames and repeat the first frame depending on the fps
    fps = 450
    frames = [imageio.imread(img) for img in images]
    frames = [frames[0]] * (fps * 3) + frames[1:-1] + [frames[-1]] * (fps * 4)

    # define the output gif file path
    video_output_path = save_dir + '{}-{}{}.mp4'.format(graph, target, str(metric_combs))

    imageio.mimwrite(video_output_path, frames, fps=fps, codec='libx264', quality=10)
    print('Done with making the Video')
