import numpy as np
import networkx as nx
import copy
import torch

from scipy.stats import wasserstein_distance_nd
from geomloss import SamplesLoss


# from deepdrawing, function used in function criterion_procrustes
def orthogonal_procrustes_torch(A, B):
    # Be clever with transposes, with the intention to save memory.
    A_device = A.device
    B_copy = B.clone().to(A_device)

    input = torch.transpose(torch.matmul(torch.transpose(B_copy, 0, 1), A), 0, 1)
    u, w, vt = torch.svd(input)
    # u, w, vt = torch.svd(torch.transpose(torch.matmul(torch.transpose(B,0,1),A),0,1))
    R = torch.matmul(u, torch.transpose(vt, 0, 1))
    scale = torch.sum(w)

    return R, scale


def criterion_procrustes(pos_1, pos_2, return_mat=  False):
    device = pos_1.device
    mtx1 = pos_1
    mtx2 = pos_2.clone().to(device)

    # translate all the data to the origin
    mtx3 = mtx1 - torch.mean(mtx1, 0)
    mtx4 = mtx2 - torch.mean(mtx2, 0)

    norm1 = torch.norm(mtx3)
    norm2 = torch.norm(mtx4)

    if norm1 == 0:
        norm1 = 1e-16
    if norm2 == 0:
        norm2 = 1e-16

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx3 = mtx3 / norm1
    mtx4 = mtx4 / norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes_torch(mtx3, mtx4)
    mtx4 = torch.matmul(mtx4, torch.transpose(R, 0, 1)) * s

    if return_mat:
        return mtx3, mtx4
    else:
        # measure the dissimilarity between the two datasets
        disparity = torch.sum((mtx3 - mtx4) ** 2)

        return disparity


# wasserstein distance
def wasserstein_test(pos_1, pos_2):

    x0, x1 = translate_scale(pos_1, pos_2)

    return wasserstein_distance_nd(x0, x1)


lossf_wasserstein = SamplesLoss(loss = 'sinkhorn', p = 2, blur = 0.001)


# sinkhorn approximation of wasserstein distance
def sinkhorn_approx(pos_1, pos_2):

    x0, x1 = translate_scale(pos_1, pos_2)

    return lossf_wasserstein(x0, x1)


def similarity(pos_1, pos_2):

    x0, x1 = translate_scale(pos_1, pos_2)
    i = 0
    n = x0.shape[0]
    all_dis = torch.tensor(0).float()
    while n > 0:
        curr_dis = torch.sqrt(torch.sum((x0[i] - x1) ** 2, 1))
        # idx of the closest
        closeness = torch.min(curr_dis, dim = 0, keepdim = False)

        all_dis += closeness.values
        close_idx = closeness.indices.item()

        x1 = torch.cat((x1[:close_idx], x1[close_idx + 1:]))

        x0 = torch.cat((x0[:0], x0[0 + 1:]))
        n = x0.shape[0]

    return all_dis


def translate_scale(x0, x1):

    # translate all the data to the origin
    mtx3 = x0 - torch.mean(x0, 0)
    mtx4 = x1 - torch.mean(x1, 0)

    norm1 = torch.norm(mtx3)
    norm2 = torch.norm(mtx4)

    if norm1 == 0:
        norm1 = 1e-16
    if norm2 == 0:
        norm2 = 1e-16

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx3 = mtx3 / norm1
    mtx4 = mtx4 / norm2

    return mtx3, mtx4


def norm_stress_torch(coords, args, stress_alpha=2):

    graph, gtds, edges = args

    with np.errstate(divide='ignore'):
        # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
        weights = torch.tensor(gtds).float() ** -stress_alpha
        weights[weights == float('inf')] = 0

    n = len(coords)
    # calculate the euclidean distances
    eucl_dis = torch.sqrt(torch.sum(((torch.unsqueeze(coords, dim=1) - coords) ** 2), 2))
    # scale the coordinates
    scal_coords = coords * (torch.nansum((eucl_dis / gtds) / torch.nansum((eucl_dis ** 2) / (gtds ** 2))))

    # compute the euclidean distances again according to scaled coordinates
    eucl_dis_new = torch.sqrt(torch.sum(((torch.unsqueeze(scal_coords, dim=1) - scal_coords) ** 2), 2))

    # compute stress
    stress_tot = torch.sum(weights * ((eucl_dis_new - gtds) ** 2)) / (n ** 2 - n)
    ns = stress_tot

    return ns


def edge_lengths_sd_torch(coords, args):

    graph, gtds, edges = args
    edge_coords = coords[edges]
    edge_coords = edge_coords.reshape(edge_coords.shape[0], 4)

    # calculate the euclidean distances
    eucl_dis = torch.sqrt(torch.sum(((edge_coords[:, 0:2] - edge_coords[:, 2:4]) ** 2), 1))

    mu = torch.mean(eucl_dis)

    # best edge length standard deviation is 0
    el = torch.sqrt(torch.mean((eucl_dis - mu) ** 2))

    return el


# taken from sgd^2 https://github.com/tiga1231/graph-drawing/blob/sgd/neural-crossing-detector.ipynb
def cross_pairs(coords, args):

    graph, gtds, edges = args

    m = len(edges)
    edge_coords = coords[edges]
    edge_coords = edge_coords.reshape(edge_coords.shape[0], 4)

    # do some matrix repetition so that we can get edge pairs later
    matrix_repeated_rows = edge_coords.repeat(m, 1)
    matrix_repeated_cols = edge_coords.repeat_interleave(m, dim=0)

    # mask to filter out self edge comparisons
    indices = torch.arange(m)
    row_indices = indices.repeat_interleave(m)
    col_indices = indices.repeat(m)
    mask = row_indices != col_indices

    filtered_rows = matrix_repeated_rows[mask]
    filtered_cols = matrix_repeated_cols[mask]

    p = torch.cat((filtered_rows, filtered_cols), dim=1)

    p1, p2, p3, p4 = p[:, :2], p[:, 2:4], p[:, 4:6], p[:, 6:]
    a = p2 - p1
    b = p3 - p4
    c = p1 - p3
    ax, ay = a[:, 0], a[:, 1]
    bx, by = b[:, 0], b[:, 1]
    cx, cy = c[:, 0], c[:, 1]

    denom = ay * bx - ax * by
    numer_alpha = by * cx - bx * cy
    numer_beta = ax * cy - ay * cx
    alpha = numer_alpha / denom
    beta = numer_beta / denom

    cross_bool = torch.logical_and(
        torch.logical_and(0 < alpha, alpha < 1),
        torch.logical_and(0 < beta, beta < 1),
    )

    return torch.sum(cross_bool)


def angular_resolution_dev(coords, args):

    graph, gtds, edges = args
    coords = np.array(coords)

    # initialize variables
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())

    all_angles = []
    # loop over all nodes
    for i in range(n):
        # only compute angles if there are at least 2 edges to a node
        curr_degree = graph.degree(nodes[i])
        smallest_angle = 360
        if curr_degree > 1:
            best_angle = 360 / curr_degree
            curr_neighbs = list(graph.neighbors(nodes[i]))

            # get the ordering and then get the angles of that specific ordering
            order_neighbs = compute_order(curr_node = nodes[i], neighbors = curr_neighbs, coords = coords)
            norm_sub = np.subtract(coords[order_neighbs, ].copy(), coords[nodes[i], ])
            sub_phi = (np.arctan2(norm_sub[:, 1:2], norm_sub[:, :1]) * 180 / np.pi)
            # get the degrees to positive 0-360
            sub_phi = ((sub_phi + 360) % 360).flatten()

            # compare the last edge with the first edge
            first = sub_phi[0]
            last = sub_phi[-1]
            angle = abs(first - last)

            # if the angle is smaller than 360 then save that as the new angle
            if angle < smallest_angle:
                smallest_angle = angle

            # now compare each consecutive edge pair to get the smallest seen angle
            while len(sub_phi) >= 2:
                first = sub_phi[0]
                second = sub_phi[1]

                # if the angle is smaller than 360 then save that as the new angle
                angle = abs(first - second)
                if angle < smallest_angle:
                    smallest_angle = angle

                sub_phi = np.delete(sub_phi, 0)

            # add the deviation of the smallest angle to the ideal angle to a list
            all_angles.append(abs((best_angle - smallest_angle) / best_angle))

    ar = np.mean(all_angles)

    return ar



"""
Function that computes the order of neighbors around a node in clockwise-order starting at 12 o'clock
Input
curr_node:      int, the integer id of the node for which we want to know the order
neighbors:      list, a list of integer ids of the neighbors of the current node
coords:         np.array or tensor, a 2xn array or tensor of x,y node coordinates

Output
neighbors:      list, the ordered list of neighbors
"""


def compute_order(curr_node, neighbors, coords):
    # get the center x and y coordinate
    center_x = coords[curr_node][0]
    center_y = coords[curr_node][1]

    # loop over all the neighbors except the last one
    for i in range(len(neighbors) - 1):
        curr_min_idx = i

        # loop over the other neighbors
        for j in range(i + 1, len(neighbors)):

            a = coords[neighbors[j]]
            b = coords[neighbors[curr_min_idx]]

            # compare the points to see which node comes first in the ordering
            if compare_points(a[0], a[1], b[0], b[1], center_x, center_y):
                curr_min_idx = j

        if curr_min_idx != i:
            neighbors[i], neighbors[curr_min_idx] = neighbors[curr_min_idx], neighbors[i]

    return neighbors


"""
Function that compares two points (nodes) to each other to determine which one comes first w.r.t. a center
Original solution from https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order

Input
a_x:            float, the x coordinate of the first node
a_y:            float, the y coordinate of the first node
b_x:            float, the x coordinate of the second node
b_y:            float, the y coordinate of the second node
center_x:       float, the x coordinate of the center node (curr_node from compute_order function)
center_y:       float, the y coordinate of the center node (curr_node from compute_order function)

Output
res:            boolean, if True then a comes before b
"""


def compare_points(a_x, a_y, b_x, b_y, center_x, center_y):
    if ((a_x - center_x) >= 0 and (b_x - center_x) < 0):
        return True

    if ((a_x - center_x) < 0 and (b_x - center_x) >= 0):
        return False

    if ((a_x - center_x) == 0 and (b_x - center_x) == 0):
        if ((a_y - center_y) >= 0 or (b_y - center_y) >= 0):
            return a_y > b_y
        return b_y > a_y

    # compute the cross product of vectors (center -> a) x (center -> b)
    det = (a_x - center_x) * (b_y - center_y) - (b_x - center_x) * (a_y - center_y)
    if (det < 0):
        return True
    if (det > 0):
        return False

    # points a and b are on the same line from the center
    # check which point is closer to the center
    d1 = (a_x - center_x) * (a_x - center_x) + (a_y - center_y) * (a_y - center_y)
    d2 = (b_x - center_x) * (b_x - center_x) + (b_y - center_y) * (b_y - center_y)

    res = d1 > d2

    return res
