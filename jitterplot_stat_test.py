import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stat
from matplotlib import rc
from statsmodels.stats.multitest import multipletests

import target_gen as ts
from metrics import similarity

# change to latex font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set(rc = {'text.usetex' : True})
sns.set_style("white", {'axes.grid': False})


"""
Script to produce the final .pdf drawings of all your morphed drawings of your graphs in your dataset. Only works if you have the coordinates of your results 
after running main.py

Change the 'graphs' variable to include more graphs
Change the 'target_names' variable to include more targets
Change the 'metric_names' variable to include more metrics
"""


if __name__ == '__main__':

    # loop over all the graphs you have in your dataset
    graphs = ['bar_albert_gen', 'polbooks', 'lnsp_131', 'gams10am', 'dwt_307']
    write_graphs = ['\emph{bar-albert}', '\emph{polbooks}', '\emph{lnsp\_131}', '\emph{gams10am}', '\emph{dwt\_307}']
    new_names_graphs = dict(zip(graphs, write_graphs))

    all_results = {}
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
        targets_names = ['cross', 'vert-lines', 'hor-lines', 'circle', 'dinosaur', 'grid']
        write_targets = [r'\texttt{X}', r'\texttt{VERT}', r'\texttt{HOR}', r'\texttt{O}', r'\texttt{DINO}', r'\texttt{GRID}']
        new_names_targets = dict(zip(targets_names, write_targets))

        # all metrics and their combinations; the combination of metric names are not to my liking so I remap them
        metric_names = ['ST', 'ELD', 'CN', 'AR', 'ST-ELD', 'ST-CN', 'ST-AR', 'CN-ELD', 'AR-ELD', 'AR-CN', 'ST-CN-ELD', 'ST-AR-ELD', 'ST-AR-CN', 'AR-CN-ELD', 'ST-AR-CN-ELD']
        write_metric_names = [r'\texttt{ST}', r'\texttt{ELD}', r'\texttt{CN}', r'\texttt{AR}', r'\texttt{ST-ELD}', r'\texttt{ST-CN}', r'\texttt{ST-AR}', r'\texttt{ELD-CN}', r'\texttt{ELD-AR}', r'\texttt{CN-AR}', r'\texttt{ST-ELD-CN}', r'\texttt{ST-ELD-AR}', r'\texttt{ST-CN-AR}', r'\texttt{ELD-CN-AR}', r'\texttt{ST-ELD-CN-AR}']
        new_names_metrics = dict(zip(metric_names, write_metric_names))

        n = G.number_of_nodes()

        # load the start coordinates X and generate the target coordinates Y
        start_pos = torch.tensor(
            np.loadtxt('data/start_coords/{}.csv'.format(graph), delimiter=',', dtype=np.float64)).float()
        np.random.seed(1337)
        targets_pos = [ts.circle_pos(n), ts.dinosaur(n), ts.lines(n=n, vert=True), ts.lines(n=n, vert=False),
                       ts.cross(n), ts.grid(n), start_pos]
        targets_names_dict = dict(zip(['circle', 'dinosaur', 'vert-lines', 'hor-lines', 'cross', 'grid', 'init'], targets_pos))

        metric_results = {}
        for metric in metric_names:

            target_results = {}
            for target in targets_names:

                target_pos = targets_names_dict[target]
                # might need to change paths here if your folder structure is different
                if os.path.exists('results/{}-{}{}-coords.csv'.format(graph, target, metric)):

                    # load the positioning, translate and scale them to [0,1]
                    og_pos = torch.tensor(np.loadtxt('results/{}-{}{}-coords.csv'.format(graph, target, metric), delimiter=',', dtype=np.float64)).float()
                    # pos = og_pos - torch.min(og_pos)
                    # pos /= torch.max(pos)

                    sim_start_target = similarity(start_pos, torch.tensor(target_pos))
                    sim_og_target = similarity(og_pos, torch.tensor(target_pos))
                    sim_value = 100 - ((sim_og_target.item() / sim_start_target.item()) * 100)

                    target_results[new_names_targets[target]] = sim_value
            metric_results[new_names_metrics[metric]] = target_results
        all_results[new_names_graphs[graph]] = metric_results

    res_list = []
    res_list_mean = []
    for graph in all_results:
        for metric in all_results[graph]:
            res_list_mean.append([graph, metric, np.mean(list(all_results[graph][metric].values()))])
            for target in all_results[graph][metric]:
                res_list.append([graph, metric, target, all_results[graph][metric][target]])

    res_df = pd.DataFrame(data = res_list, columns = ['graph', 'metric(s)', 'target shape', 'Similarity\%'])
    res_df_mean = pd.DataFrame(data = res_list_mean, columns = ['graph', 'metric(s)', 'Similarity\%'])

    plt.figure(figsize=(8, 10))
    ax = sns.stripplot(data = res_df, x = 'Similarity\%', y = 'metric(s)', hue = 'graph', jitter = 0.20, dodge = False, palette = ['red', 'black', 'yellowgreen', 'blue', 'gray'])

    # add dotted horizontal lines for visual separation
    y_locs = range(len(res_df['metric(s)'].unique()))
    for y in y_locs:
        ax.axhline(y=y + 0.5, color='gray', linestyle=':', linewidth=0.5, zorder=0)
    plt.xlabel('\emph{Similarity} in \%')
    plt.ylabel('\emph{Metrics}')
    plt.tight_layout()
    plt.savefig('results/final_simvals/{}'.format('jitter_all.pdf'), dpi = 400)
    plt.close('all')

    # plot for the average over all target shapes
    plt.figure(figsize=(8, 10))
    ax = sns.stripplot(data = res_df_mean, x = 'Similarity\%', y = 'metric(s)', hue = 'graph', jitter = 0.20, dodge = False, palette = ['red', 'black', 'yellowgreen', 'blue', 'gray'])

    # add dotted horizontal lines for visual separation
    y_locs = range(len(res_df['metric(s)'].unique()))
    for y in y_locs:
        ax.axhline(y=y + 0.5, color='gray', linestyle=':', linewidth=0.5, zorder=0)
    plt.xlabel('\emph{Similarity} in \%')
    plt.ylabel('\emph{Metric(s)}')
    plt.tight_layout()
    plt.savefig('results/final_simvals/{}'.format('jitter_mean.pdf'), dpi=400)
    plt.close('all')

    ################################################################################################################

    # to test whether some metrics are more difficult to fool than others we first have to transform the dataset to a wide one
    sub_res = res_df[['metric(s)', 'Similarity\%']]
    sub_res['observation'] = sub_res.groupby('metric(s)').cumcount()

    # pivot
    wide_df = sub_res.pivot(index='observation', columns='metric(s)', values='Similarity\%')

    # friedman test shows metrics have different distribution(s)
    statistic, p = stat.friedmanchisquare(*[wide_df[q] for q in wide_df.columns])

    # now we test one metric with every other metric
    # AR, ELD, ST-ELD, ST-AR, ELD-AR
    metrics_test = ['AR', 'ELD', 'ST-ELD', 'ST-AR', 'ELD-AR']
    for m in metrics_test:
        print('\nTesting for metric (combination): ' + m)
        p_values = []
        other_metrics = []
        for q in wide_df.columns:  # skip Q1
            if q != r'\texttt{' + m + '}':
                statistic, p = stat.wilcoxon(wide_df[r'\texttt{' + m + '}'], wide_df[q], alternative = 'greater')
                p_values.append(p)
                metric_name = q.replace('\\texttt{', '')
                metric_name = metric_name.replace('}', '')
                other_metrics.append(metric_name)

        # correct the p-values
        rej, p_adj, _, _ = multipletests(p_values, method='holm', alpha = 0.05)
        print('Reject null-hypotheses?')
        for i in range(len(other_metrics)):
            print(other_metrics[i] + ' ' + str(rej[i]))
            # print(other_metrics[i] + ' uncorrected: ' + str(p_values[i] < 0.05))


    # create a significance matrix to visualize
    # reorder the columns
    wide_df = wide_df.loc[:, write_metric_names]
    n_metrics = len(wide_df.columns)
    metrics = list(wide_df.columns)

    p_matrix = np.ones((n_metrics, n_metrics))
    p_vals = []
    pairs = []
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i != j:
                statistic, p = stat.wilcoxon(wide_df.iloc[:, i], wide_df.iloc[:, j], alternative = 'greater')
                p_matrix[i, j] = p
                p_vals.append(p)
                pairs.append([i, j])

    # correct for multiple testing
    rej, p_adj, _, _ = multipletests(p_vals, method='holm', alpha = 0.05)

    # use the adjusted p values
    sig_matrix = np.full((n_metrics, n_metrics), False)
    for i in range(len(pairs)):
        sig_matrix[pairs[i][0], pairs[i][1]] = p_adj[i] < 0.05

    # create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = {True: 'yellow', False: 'lightgray'}
    color_matrix = np.vectorize(cmap.get)(sig_matrix)

    for i in range(n_metrics):
        for j in range(n_metrics):
            if i == j:
                color_matrix[i, j] = 'white'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color = color_matrix[i, j]))

    # add labels
    ax.set_xlim(0, n_metrics)
    ax.set_ylim(0, n_metrics)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(n_metrics) + 0.5)
    ax.set_xticklabels(metrics, rotation = 45)
    ax.set_yticks(np.arange(n_metrics) + 0.5)
    ax.set_yticklabels(metrics)

    # add dividers between cells
    for i in range(1, n_metrics):
        ax.axhline(i, color='black', linestyle=':', linewidth=0.5)
        ax.axvline(i, color='black', linestyle=':', linewidth=0.5)

    # get rid of the major lines and save
    ax.tick_params(which = 'major', length=0)
    ax.invert_yaxis()
    ax.set_title("Sign. Matrix Metric Comparison")
    plt.tight_layout()
    plt.savefig('results/final_simvals/sig_matrix_metrics.pdf', dpi = 400)
    plt.close('all')


    ################################################################################################################
    # to test whether some target shapes are more difficult to fool into than others we first have to transform the dataset to a wide one
    sub_res = res_df[['target shape', 'Similarity\%']]
    sub_res['observation'] = sub_res.groupby('target shape').cumcount()

    # pivot
    wide_df = sub_res.pivot(index='observation', columns='target shape', values='Similarity\%')

    # friedman test shows metrics have different distribution(s)
    statistic, p = stat.friedmanchisquare(*[wide_df[q] for q in wide_df.columns])

    # create a significance matrix to visualize
    # reorder the columns
    wide_df = wide_df.loc[:, write_targets]
    n_targets = len(wide_df.columns)
    targets = list(wide_df.columns)

    p_matrix = np.ones((n_targets, n_targets))
    p_vals = []
    pairs = []
    for i in range(n_targets):
        for j in range(n_targets):
            if i != j:
                statistic, p = stat.wilcoxon(wide_df.iloc[:, i], wide_df.iloc[:, j], alternative = 'greater')
                p_matrix[i, j] = p
                p_vals.append(p)
                pairs.append([i, j])

    # correct for multiple testing
    rej, p_adj, _, _ = multipletests(p_vals, method='holm', alpha = 0.05)

    # use the adjusted p values
    sig_matrix = np.full((n_targets, n_targets), False)
    for i in range(len(pairs)):
        sig_matrix[pairs[i][0], pairs[i][1]] = p_adj[i] < 0.05

    # create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8/15*6, 8/15*6))
    cmap = {True: 'yellow', False: 'lightgray'}
    color_matrix = np.vectorize(cmap.get)(sig_matrix)

    for i in range(n_targets):
        for j in range(n_targets):
            if i == j:
                color_matrix[i, j] = 'white'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color = color_matrix[i, j]))

    # add labels
    ax.set_xlim(0, n_targets)
    ax.set_ylim(0, n_targets)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(n_targets) + 0.5)
    ax.set_xticklabels(targets, rotation = 45)
    ax.set_yticks(np.arange(n_targets) + 0.5)
    ax.set_yticklabels(targets)

    # add dividers between cells
    for i in range(1, n_targets):
        ax.axhline(i, color='black', linestyle=':', linewidth=0.5)
        ax.axvline(i, color='black', linestyle=':', linewidth=0.5)

    # get rid of the major lines and save
    ax.tick_params(which = 'major', length=0)
    ax.invert_yaxis()
    ax.set_title("Sign. Matrix Target Comparison")
    plt.tight_layout()
    plt.savefig('results/final_simvals/sig_matrix_targets.pdf', dpi = 400)
    plt.close('all')

    ################################################################################################################
    # now we test for differences between graphs
    sub_res = res_df[['graph', 'Similarity\%']]
    sub_res['observation'] = sub_res.groupby('graph').cumcount()

    # pivot
    wide_df = sub_res.pivot(index='observation', columns='graph', values='Similarity\%')

    n_graphs = len(wide_df.columns)
    graphs = list(wide_df.columns)

    p_matrix = np.ones((n_graphs, n_graphs))
    p_vals = []
    pairs = []
    for i in range(n_graphs):
        for j in range(n_graphs):
            if i != j:
                statistic, p = stat.wilcoxon(wide_df.iloc[:, i], wide_df.iloc[:, j], alternative = 'greater')
                p_matrix[i, j] = p
                p_vals.append(p)
                pairs.append([i, j])

        # correct for multiple testing
    rej, p_adj, _, _ = multipletests(p_vals, method='holm', alpha=0.05)

    # use the adjusted p values
    sig_matrix = np.full((n_graphs, n_graphs), False)
    for i in range(len(pairs)):
        sig_matrix[pairs[i][0], pairs[i][1]] = p_adj[i] < 0.05

    # create a matplotlib figure
    fig, ax = plt.subplots(figsize=(8/15*6, 8/15*6))
    cmap = {True: 'yellow', False: 'lightgray'}
    color_matrix = np.vectorize(cmap.get)(sig_matrix)

    for i in range(n_graphs):
        for j in range(n_graphs):
            if i == j:
                color_matrix[i, j] = 'white'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_matrix[i, j]))

    # add labels
    ax.set_xlim(0, n_graphs)
    ax.set_ylim(0, n_graphs)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(n_graphs) + 0.5)
    ax.set_xticklabels(graphs, rotation=45)
    ax.set_yticks(np.arange(n_graphs) + 0.5)
    ax.set_yticklabels(graphs)

    # add dividers between cells
    for i in range(1, n_graphs):
        ax.axhline(i, color='black', linestyle=':', linewidth=0.5)
        ax.axvline(i, color='black', linestyle=':', linewidth=0.5)

    # get rid of the major lines and save
    ax.tick_params(which='major', length=0)
    ax.invert_yaxis()
    ax.set_title("Sign. Matrix Graph Comparison")
    plt.tight_layout()
    plt.savefig('results/final_simvals/sig_matrix_graphs.pdf', dpi=400)
    plt.close('all')

