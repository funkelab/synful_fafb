import pandas as pd
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import sys
import json
import os
import csv
import numpy as np
from matplotlib import pyplot as plt


def get_adjmatrix(source_ids, target_ids, adj_matrix):
    adj_matrix = pd.read_csv(adj_matrix)
    if 'sources' in adj_matrix.keys():
        adj_matrix = adj_matrix.set_index('sources')
    adj_matrix = adj_matrix.reindex(source_ids)
    adj_matrix = adj_matrix[[str(id) for id in target_ids]]
    print(adj_matrix.shape, 'shape of adj matrix')
    print(np.sum(np.array(adj_matrix)), 'number of total connections')

    return adj_matrix


def plot_pred_vs_gt(gt, pred, outputdir, vmax=None, freqmax=None):
    sns.set_style(None)
    sns.set(font_scale=2)
    plt.figure(figsize=(12, 10))
    sns.set_style("ticks")

    x = pred[(gt != 0) | (pred != 0)].flatten()
    y = gt[(gt != 0) | (pred != 0)].flatten()

    tuplelist = [*zip(x, y)]
    freq = [tuplelist.count(item) for item in tuplelist]
    if freqmax is None:
        freqmax = np.max(freq)

    plt.scatter(x, y, c=freq, s=30,
                cmap=sns.cubehelix_palette(50, start=.5, rot=-.75,
                                           as_cmap=True),
                norm=colors.LogNorm(vmin=np.min(freq), vmax=freqmax))
    if vmax is None:
        vmax = max(np.max(x), np.max(y)) + 10
    plt.plot(range(np.int(vmax) + 10), range(np.int(vmax) + 10), '--',
             color='black', linewidth=0.5)
    plt.xlim((-5, vmax))
    plt.ylim((-5, vmax))
    plt.gca().set_aspect(1.0)
    cbar = plt.colorbar(format='%i')
    cbar.set_label('Neuron pair frequency')
    plt.xlabel('Predicted synapse count')
    plt.ylabel('Ground-truth synapse count')
    sns.despine()
    plt.savefig(outputdir + 'gt_versus_pred_count.png', dpi=300,
                transparent=True)


def plot_connectivity_matrix_scatter(mat, vmax=None, vmin=None, cmap=None):
    # sns.set_style(None)
    sns.set(font_scale=3)
    plt.figure(figsize=(20, 20))
    sns.set_style("ticks")
    # sns.set_style('darkgrid')
    plt.gca().set_aspect(1.0)
    coords = np.where(mat != 0)
    colors = mat[mat != 0].flatten()
    plt.scatter(coords[1], coords[0], c=colors, s=10, cmap=cmap, vmax=vmax,
                vmin=vmin)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel('Pre-synaptic')
    plt.xlabel('Post-synaptic')
    plt.xlim([-1, mat.shape[1] + 1])
    plt.ylim([-1, mat.shape[0] + 1])
    ax = plt.gca()


def plot_adjmatrices(gt, pred, outputdir, vmax=30):
    sns.set_palette('bright')
    current_palette = sns.color_palette()
    sns.palplot(current_palette)

    green = current_palette[2]
    red = current_palette[3]
    lightblue = current_palette[-1]

    n_bin = 120
    colors = [green, red, lightblue]
    cm = LinearSegmentedColormap.from_list(
        'fp_fn_color', colors, N=n_bin)

    # Plot Ground Truth
    plot_connectivity_matrix_scatter(gt, cmap=cm, vmax=vmax)
    plt.title('Ground-truth connectivity matrix')
    cbar = plt.colorbar(fraction=0.013, pad=0.01)
    labels = [int(item) for item in cbar.get_ticks()]
    labels[-1] = f'>{vmax}'
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('# synaptic connections')
    sns.despine()
    plt.savefig(outputdir + 'gt.png', dpi=300, transparent=True,
                bbox_inches='tight')

    # Plot Pred
    plot_connectivity_matrix_scatter(pred, cmap=cm, vmax=vmax)
    plt.title('Predicted connectivity matrix')
    cbar = plt.colorbar(fraction=0.013, pad=0.01)
    labels = [int(item) for item in cbar.get_ticks()]
    labels[-1] = f'>{vmax}'
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('# synaptic connections')
    sns.despine()
    plt.savefig(outputdir + 'pred.png', dpi=300, transparent=True,
                bbox_inches='tight')

    # Plot Difference
    difference = gt - pred
    n_bin = 120
    colors = ['red', 'white', 'blue']
    cm = LinearSegmentedColormap.from_list(
        'fp_fn_color', colors, N=n_bin)
    plot_connectivity_matrix_scatter(difference, vmin=-22.5, vmax=22.5, cmap=cm)
    cbar = plt.colorbar(fraction=0.013, pad=0.01)
    plt.title('Difference')
    plt.ylabel('Pre-synaptic')
    plt.xlabel('Post-synaptic')
    cbar.set_label('# synaptic connections')
    sns.despine()
    plt.savefig(outputdir + 'difference_scatter.png', dpi=300,
                transparent=True, bbox_inches='tight')


def eval_pairwise_connection(gt_con, pred_con, connection_thr=5):
    fn = np.sum((pred_con < connection_thr) & (gt_con >= connection_thr))
    tp = np.sum((gt_con >= connection_thr) & (pred_con >= connection_thr))
    fp = np.sum((gt_con < connection_thr) & (pred_con >= connection_thr))
    tn = np.sum(
        (gt_con < connection_thr) & (pred_con < connection_thr) & (gt_con != 0))
    accuracy = (tp + tn) / float((tp + tn + fp + fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * (precision * recall) / (precision + recall)
    result = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'fscore': fscore,
        'edge_threshold': connection_thr,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    return result


def __evaluate_connectivity(gt_adj_matrix, pred_adj_matrix, outputdir=None,
                            vmax=None):
    if outputdir is not None:
        plot_pred_vs_gt(np.array(gt_adj_matrix), np.array(pred_adj_matrix),
                        outputdir, vmax=vmax)
        plot_adjmatrices(np.array(gt_adj_matrix), np.array(pred_adj_matrix),
                         outputdir)
        np.savetxt(outputdir + '/pred_adj_matrix.csv',
                   np.array(pred_adj_matrix), fmt='%i', delimiter=",")
        np.savetxt(outputdir + '/gt_adj_matrix.csv', np.array(gt_adj_matrix),
                   fmt='%i', delimiter=",")

    thresholds = range(1, 30)
    results = []
    for threshold in thresholds:
        res = eval_pairwise_connection(np.array(gt_adj_matrix),
                                       np.array(pred_adj_matrix),
                                       connection_thr=threshold)
        results.append(res)
    return pd.DataFrame(results)


def get_adj_matrix(pred_syn, source_ids, target_ids, score_thr=None):
    df_syn = pd.read_json(pred_syn)
    if score_thr is not None:
        df_syn = df_syn[(df_syn['id_skel_pre'].notnull()) & (
            df_syn['id_skel_post'].notnull()) & (
                                df_syn['score'] >= score_thr)]
    else:
        df_syn = df_syn[(df_syn['id_skel_pre'].notnull()) & (
            df_syn['id_skel_post'].notnull())]
    df_syn = list(zip(df_syn['id_skel_pre'], df_syn['id_skel_post']))

    adj_matrix = np.zeros((len(source_ids), len(target_ids)))
    source_to_index = {}
    for ii, id in enumerate(source_ids):
        source_to_index[id] = ii

    target_to_index = {}
    for ii, id in enumerate(target_ids):
        target_to_index[id] = ii

    other_cons = 0
    for u, v in df_syn:
        u = source_to_index.get(int(u), None)
        v = target_to_index.get(int(v), None)
        if u is not None and v is not None and not u == v:
            adj_matrix[u, v] += 1
        else:
            other_cons += 1

    adj_matrix = pd.DataFrame(adj_matrix, columns=target_ids, index=source_ids)
    print('loaded {} of synapses'.format(len(df_syn)))
    print('in pred adj matrix total connections: {}'.format(
        np.sum(np.array(adj_matrix))))
    return adj_matrix


def __csv_to_list(csvfilename, column):
    with open(csvfilename) as csvfile:
        data = list(csv.reader(csvfile))
    col_list = []
    for ii in range(0, len(data)):
        row = data[ii]
        col_list.append(int(row[column]))
    return col_list


def evaluate_connectivity(source_ids, target_ids, pred_syn, gt_adj=None,
                          gt_db_name=None,
                          gt_db_col=None, gt_db_host=None, score_thr=0,
                          outputdir=None,
                          add_synaptic_cleft_score=False,
                          synful_method='synfulv01', dataset=''):
    source_ids = __csv_to_list(source_ids, 0)
    target_ids = __csv_to_list(target_ids, 0)
    vmax = None

    print('---- Loading predicted adj matrix')
    pred_adj = get_adj_matrix(pred_syn,
                              source_ids, target_ids, score_thr=score_thr)

    print('---Loading GT matrix')
    if gt_adj is not None:
        gt_adj = get_adjmatrix(source_ids, target_ids, gt_adj)
    else:
        assert gt_db_name is not None
        gt_adj = get_adj_matrix(gt_db_name, gt_db_col, gt_db_host,
                                source_ids, target_ids)

    results = __evaluate_connectivity(gt_adj, pred_adj, outputdir, vmax=vmax)
    results['add_synaptic_cleft_score'] = add_synaptic_cleft_score
    results['synful_method'] = synful_method
    results['dataset'] = dataset
    results.to_csv(outputdir + 'edge_accuracy.csv', index=False)
    print(results[results.edge_threshold == 5])


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)
    outputdir = config['outputdir']
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        print('creating {}'.format(outputdir))

    evaluate_connectivity(**config)
