import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

import criteria_torch_np as CNP
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore")


def perturb_torch(coords):

    # get a random number of points
    n_points = np.random.randint(1, int(len(coords) / 15))
    # get a random selection of nodes
    rand_idx = np.random.choice(len(coords), n_points)
    # shift these nodes a tiny bit

    new_coords = coords.clone()
    new_coords[rand_idx] += (torch.rand(n_points, 2) - 0.5) / 25
    # new_coords[np.random.randint(1, len(coords))] += np.random.rand(2) / 10

    return new_coords


def sim_anneal_torch(x0, args, start_temp, max_N, abs_diff=0.0075, wd_diff = 0.005):

    tar_coords, tar_value, qm_func, G = args
    new_sol = x0.clone()
    # curr_loss_WD = torch.tensor(CNP.wasserstein_test(x0, tar_coords))
    curr_loss_WD = torch.tensor(CNP.stat_metric(x0, tar_coords))
    best_loss_wd = curr_loss_WD.clone()
    global_best_sol = new_sol.clone()
    curr_loss_qm = 1
    temp_sched = np.geomspace(start_temp, 0.001, num = max_N)
    gtds = nx.floyd_warshall_numpy(G)
    print('\nStart with a Wasserstein Distance of: ' + str(round(curr_loss_WD.item(), 4)))

    stop_approx = True
    approx_diff = 0.0001

    pbar = tqdm(total = max_N)
    print('\nUsing the Sinkhorn algorithm at the start')
    start = time.time()
    i = 0

    while True:

        curr_temp = temp_sched[i]
        while True:
            new_coords = perturb_torch(new_sol)

            if stop_approx:
                # new_loss_WD = torch.tensor(CNP.wasserstein_test(new_coords, tar_coords))
                new_loss_WD = torch.tensor(CNP.stat_metric(new_coords, tar_coords))
            else:
                # new_loss_WD = torch.tensor(CNP.wasserstein_test2(new_coords, tar_coords))
                new_loss_WD = torch.tensor(CNP.stat_metric(new_coords, tar_coords))

            if new_loss_WD < curr_loss_WD or curr_temp > np.random.rand(1):
                break

        if not stop_approx:
            if (torch.abs(new_loss_WD) <= approx_diff) or ((time.time() - start) >= 300) :
                stop_approx = True
                print('\nNow computing the exact Wasserstein Distance, starting loop')
                curr_loss_WD = torch.tensor(CNP.wasserstein_test(new_coords, tar_coords))
        else:
            i += 1
            pbar.update(i - pbar.n)

        if qm_func.__name__ == 'norm_stress_torch':
            qm_loss = qm_func(new_coords, gtds)
        elif qm_func.__name__ == 'edge_lengths_sd_torch':
            qm_loss = qm_func(new_coords, np.array(G.edges()))
        elif qm_func.__name__ == 'angular_resolution_dev':
            qm_loss = torch.tensor(qm_func(np.array(new_coords), gtds))
        elif qm_func.__name__ == 'cross_pairs':
            qm_loss = qm_func(new_coords, np.array(G.edges()))

        if torch.abs(tar_value - qm_loss) <= abs_diff:
            new_sol = new_coords.clone()
            curr_loss_WD = new_loss_WD
            curr_loss_qm = qm_loss
            pbar.set_description("Best WD: " + str(round(best_loss_wd.item(), 4)) + "| Current WD: " + str(round(curr_loss_WD.item(), 4)) + "| curr loss: " + str(round(curr_loss_qm.item(), 4)))

            if curr_loss_WD <= best_loss_wd:
                global_best_sol = new_sol.clone()
                best_loss_wd = curr_loss_WD

        if i >= max_N:
            break
        if i > 1000 and curr_loss_WD <= wd_diff:
            print('\nConverged Early')
            break

        if time.time() - start >= 3600:
            print('\nConverged Early')
            break

    # final_WD = torch.tensor(CNP.wasserstein_test(new_coords, tar_coords))
    # final_WD = torch.tensor(CNP.stat_metric(new_sol, tar_coords))
    final_WD = torch.tensor(CNP.stat_metric(global_best_sol, tar_coords))
    final_diff = curr_loss_qm - tar_value

    return {'coords': global_best_sol, 'WD': final_WD, 'qm_diff': final_diff, 'qm_og' : tar_value, 'qm_new' : curr_loss_qm}


def save_res(coords, G, graph_name, metric_name, title):

    plt.title(title, fontsize = 10)
    pos = np.array(coords)
    pos_G = {k: list(pos[k]) for k in G.nodes()}
    pos_G = {k : [pos_G[k][0].item(), pos_G[k][1].item()] for k in pos_G}
    nx.draw(G, with_labels=False, pos=pos_G, node_size=16, edge_color = ['lightblue'], width = 1)
    plt.savefig('results/{}{}.png'.format(graph_name, metric_name))
    plt.close('all')

    # save the coordinates
    np.savetxt('results/{}{}-coords.csv'.format(graph_name, metric_name), pos, delimiter = ',')


# edl = np.loadtxt('data/edgelists/bar_albert_gen-edgelist.csv', delimiter = ',').astype(np.int64)
# temp_G = nx.Graph()
# edgelist = []
# for e in edl:
#     edgelist.append([e[0].item(), e[1].item()])
# temp_G.add_edges_from(edgelist)
# temp_G.remove_edges_from(nx.selfloop_edges(temp_G))
# G = nx.convert_node_labels_to_integers(temp_G)
# G.remove_edges_from(nx.selfloop_edges(G))
#
# target_pos = TG.circle_pos(G.number_of_nodes())
#
# og_pos = torch.tensor(np.loadtxt('data/start_coords/bar_albert_gen.csv', delimiter = ',', dtype = np.float64)).float()
#
#
# qm_target = CNP.edge_lengths_sd_torch(og_pos, np.array(G.edges()))
# test2 = sim_anneal_torch(x0 = og_pos, args = [torch.tensor(target_pos).float(), qm_target, CNP.edge_lengths_sd_torch, G], start_temp = 0.4, max_N = 500, abs_diff = 0.0075, ps_diff = 0)
#
# # qm_target = CNP.cross_pairs(og_pos, np.array(gd.G.edges()))
# # test2 = sim_anneal_torch(x0 = np.array(og_pos), args = [torch.tensor(target_pos).float(), qm_target, CNP.cross_pairs, gd], start_temp = 0.4, max_N = 400000, abs_diff = 5, ps_diff = 0.0005)
#
# # qm_target = CNP.norm_stress_torch(og_pos, gd.gtds)
# # test2 = sim_anneal_torch(x0 = og_pos, args = [torch.tensor(target_pos).float(), qm_target, CNP.norm_stress_torch, gd], start_temp = 0.4, max_N = 2000, abs_diff = 0.0075, ps_diff = 0.0)
#
# print('Procrustes Statistic: ' + str(round(test2['ps'].item(), 4)))
# curr_qm_val = CNP.edge_lengths_sd_torch(test2['coords'], np.array(G.edges()))
# print('Target QM val: ' + str(round(qm_target.item(), 4)))
# print('Curr QM val: ' + str(round(curr_qm_val.item(), 4)))
# print('QM diff: ' + str(round(np.abs(test2['qm_diff'].item()), 4)))
#
# title = 'Original ST: ' + str(round(qm_target.item(), 4)) + ' | current ST: ' + str(round(curr_qm_val.item(), 4)) + ' | Wasserstein2: ' + str(round(test2['ps'].item(), 4))
# save_res(test2['coords'], G, graph_name = 'test', metric_name = 'eld', title = title)

