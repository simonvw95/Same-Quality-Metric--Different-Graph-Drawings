## custom
from sgd2.utils import utils
import sgd2.criteria as C
from sgd2.gd2 import GD2
import sgd2.utils.weight_schedule as ws

## third party
import networkx as nx

### numeric
import numpy as np
import torch


## sys
import random
import pickle as pkl

device = 'cpu'

seed = 2337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

graph_name = 'dwt_307'
max_iter = int(5e4)
mat_dir = 'sgd2/input_graphs/SuiteSparse Matrix Collection'
G = utils.load_mat(f'{mat_dir}/{graph_name}.mat')
G.remove_edges_from(nx.selfloop_edges(G))

# criteria = ['stress']#, 'ideal_edge_length', 'aspect_ratio']
# criteria = ['ps']
# criteria = ['sup_stress', 'ps']
criteria = ['sup_stress']
criteria_weights = dict(
    sup_stress=ws.SmoothSteps([0, max_iter * 0.2, max_iter * 0.6, max_iter], [0.4, 0.5, 0.6, 0.8]))  # ,
# ps=ws.SmoothSteps([max_iter/4, max_iter], [1, 0.05]))
# stress=ws.SmoothSteps([max_iter/4, max_iter], [1, 0.05]))#,
# ps=ws.SmoothSteps([max_iter/4, max_iter], [1, 0.05]))#,
#    stress_ps=ws.SmoothSteps([max_iter/4, max_iter], [1, 0.05]))#,
#     ideal_edge_length=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.2, 0]),
#     aspect_ratio=ws.SmoothSteps([0, max_iter*0.2, max_iter*0.6, max_iter], [0, 0, 0.5, 0]),
# )
criteria = list(criteria_weights.keys())

# plot_weight(criteria_weights, max_iter)
# plt.close()


sample_sizes = dict(
    sup_stress=16,
    ps=16,
    stress=16,
    ideal_edge_length=16,
    sup_el=16,
    neighborhood_preservation=16,
    crossings=128,
    crossing_angle_maximization=64,
    aspect_ratio=max(128, int(len(G) ** 0.5)),
    angular_resolution=16,
    vertex_resolution=max(256, int(len(G) ** 0.5)),
    gabriel=64,
)
sample_sizes = {c: sample_sizes[c] for c in criteria}

circle_graph = nx.cycle_graph(len(G))
pos_circle = nx.kamada_kawai_layout(circle_graph)
target_pos = torch.tensor([[float(pos_circle[k][0]), float(pos_circle[k][1])] for k in pos_circle])

gd = GD2(G, target_pos=target_pos)
with open('results/dtw_307-init.pkl', 'rb') as fp:
    gd_file = pkl.load(fp)

og_pos = gd_file['pos'].detach()
# gd.qm_target = C.ideal_edge_length(og_pos, gd.G, gd.k2i)
gd.qm_target = C.stress(og_pos, gd.D, gd.W)

result = gd.optimize(
    criteria_weights=criteria_weights,
    sample_sizes=sample_sizes,

    # evaluate='all',
    evaluate=set(criteria),

    max_iter=max_iter,
    time_limit=3600,  ##sec

    evaluate_interval=max_iter, evaluate_interval_unit='iter',
    vis_interval=-1, vis_interval_unit='sec',

    clear_output=True,
    grad_clamp=20,
    criteria_kwargs=dict(
        aspect_ratio=dict(target=[1, 1]),
    ),
    optimizer_kwargs=dict(mode='Adam', lr=0.01),
    # optimizer_kwargs = dict(mode='SGD', lr=2),
    scheduler_kwargs=dict(verbose=True),
)

pos = gd.best_pos.detach()
print('current st')
print(C.stress(pos, gd.D, gd.W))
print('target st')
print(gd.qm_target)
print('procrustes statistic')
print(C.criterion_procrustes(pos, target_pos))
