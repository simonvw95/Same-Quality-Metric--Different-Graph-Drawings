import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

from tqdm import tqdm

import metrics as qm

np.seterr(divide = 'ignore', invalid = 'ignore')
warnings.filterwarnings("ignore")

# DEPRECATED CODE, USE sim_anneal_mult.py


def perturb_torch(coords):

    # get a random number of points
    n_points = np.random.randint(1, int(len(coords) / 15))
    # get a random selection of nodes
    rand_idx = np.random.choice(len(coords), n_points)

    # shift these nodes a tiny bit
    new_coords = coords.clone()
    new_coords[rand_idx] += (torch.rand(n_points, 2) - 0.5) / 25

    return new_coords


def sim_anneal_torch(x0, args, args_qm, start_temp, max_N, abs_diff = 0.0075):

    # args for simulated annealing
    tar_coords, tar_value, qm_func = args
    # copy the starting coordinates and compute the similarity
    new_sol = x0.clone()
    curr_loss_sim = torch.tensor(qm.similarity(x0, tar_coords))
    best_loss_sim = curr_loss_sim.clone()
    global_best_sol = new_sol.clone()
    curr_loss_qm = 1

    # schedule for the temperature
    temp_sched = np.geomspace(start_temp, 0.001, num = max_N)

    pbar = tqdm(total = max_N)
    print('\nStart with a Similarity of: ' + str(round(curr_loss_sim.item(), 4)))
    start = time.time()
    i = 0

    while True:

        # set the current temperature
        curr_temp = temp_sched[i]

        while True:

            # generate new coordinates by perturbing them
            new_coords = perturb_torch(new_sol)

            new_loss_sim = torch.tensor(qm.similarity(new_coords, tar_coords))

            # accept new coordinates if it is better w.r.t. similarity or if the current temp is better than rand value
            if new_loss_sim < curr_loss_sim or curr_temp > np.random.rand(1):
                break

        i += 1
        pbar.update(i - pbar.n)

        # compute the quality metric value of the new coordinates
        qm_loss = qm_func(new_coords, args_qm)

        # if the quality metric value is within the acceptable range (abs_diff) then we accept the new coordinates and replace the old coordinates
        if torch.abs(tar_value - qm_loss) <= abs_diff:
            new_sol = new_coords.clone()
            curr_loss_sim = new_loss_sim
            curr_loss_qm = qm_loss

            # update the tqdm bar
            pbar.set_description("Best Similarity: " + str(round(best_loss_sim.item(), 4)) + "| Current Similarity: " + str(round(curr_loss_sim.item(), 4)) + "| curr qm loss: " + str(round(curr_loss_qm.item(), 4)))

            # set the globally best solution found so far (sim annealing might explore to a worse solution similarity wise)
            if curr_loss_sim <= best_loss_sim:
                global_best_sol = new_sol.clone()
                best_loss_sim = curr_loss_sim

        # break the loop once we reach N iterations
        if i >= max_N:
            break

        # break the loop if it takes too long
        if time.time() - start >= 3600:
            print('\nConverged Early')
            break

    final_sim = torch.tensor(qm.similarity(global_best_sol, tar_coords))
    final_diff = curr_loss_qm - tar_value

    return {'coords': global_best_sol, 'sim': final_sim, 'qm_diff': final_diff, 'qm_og' : tar_value, 'qm_new' : curr_loss_qm}


def save_res(coords, G, graph_name, metric_name, title):

    # set the title
    plt.title(title, fontsize = 10)
    pos = np.array(coords)

    # convert nodes to dictionary for networkx
    pos_G = {k: list(pos[k]) for k in G.nodes()}
    pos_G = {k : [pos_G[k][0].item(), pos_G[k][1].item()] for k in pos_G}
    nx.draw(G, with_labels = False, pos = pos_G, node_size = 16, edge_color = ['lightblue'], width = 1)
    plt.savefig('results/{}{}.png'.format(graph_name, metric_name))
    plt.close('all')

    # save the coordinates
    np.savetxt('results/{}{}-coords.csv'.format(graph_name, metric_name), pos, delimiter = ',')
