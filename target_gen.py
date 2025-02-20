import numpy as np
import networkx as nx


def circle_pos(n):

    circle_graph = nx.cycle_graph(n)
    # pos_circle = nx.forceatlas2_layout(circle_graph)
    # Kamada Kawi instead of FA2 since FA2 sometimes twists circles..
    pos_circle = nx.kamada_kawai_layout(circle_graph)
    target_pos = np.array([[float(pos_circle[k][0]), float(pos_circle[k][1])] for k in pos_circle])

    return target_pos


def lines(n, vert = True):

    leftover_remove = 3 - (n % 3)

    if leftover_remove != 3:
        add = leftover_remove
    else:
        add = 0

    n_line = int((n + add) / 3)

    if vert:
        vert_range = np.linspace(0.2, 0.8, n_line)
        hor_range = np.array([0.2, 0.5, 0.8])
    else:
        hor_range = np.linspace(0.2, 0.8, n_line)
        vert_range = np.array([0.2, 0.5, 0.8])

    coords = []
    for x in hor_range:
        for y in vert_range:
            coords.append([x.item(), y.item()])

    select_idxs = np.random.choice(range(len(coords)), n, replace = False)

    pos = np.array(coords)[select_idxs]

    return pos


def cross(n):

    add = n % 2

    n_line = int((n + add) / 2)

    hor_range_1 = np.linspace(0.2, 0.8, n_line)
    ver_range_1 = np.linspace(0.2, 0.8, n_line)

    hor_range_2 = np.linspace(0.2, 0.8, n_line)
    ver_range_2 = np.linspace(0.8, 0.2, n_line)

    coords = []
    for i in range(n_line):
        coords.append([hor_range_1[i], ver_range_1[i].item()])
        coords.append([hor_range_2[i], ver_range_2[i].item()])

    if add:
        rand_idxs = np.random.randint(0, n, size = 1)
        pos = np.delete(np.array(coords), rand_idxs, axis=0)
    else:
        pos = np.array(coords)

    return pos


def grid(n):

    rows = int(np.sqrt(n))
    cols = rows if rows * rows == n else rows + 1
    if rows * cols < n:
        rows += 1

    hor_range = np.linspace(0.2, 0.8, cols)
    ver_range = np.linspace(0.2, 0.8, rows)

    coords = []
    for i in range(rows):
        for j in range(cols):
            if len(coords) < n:
                coords.append([hor_range[j], ver_range[i]])

    pos = np.array(coords)

    return pos


def dinosaur(n):

    dino_pos = np.loadtxt('data/start_coords/dino-init.csv', delimiter = ',')
    curr_n = dino_pos.shape[0]

    # translate all the data to the origin
    new_pos = dino_pos - np.min(dino_pos)

    norm1 = np.linalg.norm(new_pos)

    if norm1 == 0:
        norm1 = 1e-16

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    dino_pos = new_pos / norm1

    # remove nodes
    if n < curr_n:
        select_idxs = np.random.choice(range(len(dino_pos)), n, replace = False)
        pos = dino_pos[select_idxs]
    # add nodes
    elif n > curr_n:
        repl = False
        if n - curr_n > 0:
            repl = True

        select_idxs = np.random.choice(range(len(dino_pos)), n - curr_n, replace = repl)
        rand_jitter = (np.random.rand(len(select_idxs), 2) - 0.5) / 200
        added_coords = dino_pos[select_idxs] + rand_jitter
        pos = np.concat((dino_pos, added_coords))
    else:
        pos = dino_pos

    return pos


def bar_alb(n):

    isconn = False
    np.random.seed(1535)
    while not isconn:
        G = nx.dual_barabasi_albert_graph(n = n, m1 = 3, m2 = 1, p = 0.15)
        G.remove_edges_from(nx.selfloop_edges(G))
        isconn = nx.is_connected(G)

    G = nx.convert_node_labels_to_integers(G)
    pos_dict = nx.forceatlas2_layout(G, max_iter = 5000)

    pos = np.array([[float(pos_dict[k][0]), float(pos_dict[k][1])] for k in pos_dict])

    return pos
