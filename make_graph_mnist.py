import os, shutil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.datasets import mnist
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def _grid_coordinates_from_img(in_img, threshold):
    """
    Returns 2D coordinates for a square grid of equally spaced nodes.
    :param side: int, the side of the grid (i.e., the grid has side * side nodes).
    :return: np.array of shape (side * side, 2).
    """
    x = np.linspace(0, 1, in_img.shape[0], dtype=np.float32)
    y = np.linspace(0, 1, in_img.shape[1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # use reval function for flatten
    z = np.stack([
        xx[in_img>threshold].ravel(),
        yy[in_img>threshold].ravel(),
        in_img[in_img>threshold].ravel(),
    ], -1)
    z = z[np.argsort(-z[:, 2]), :] # sort by pixel value
    return z

def _get_adj_from_data(X, k, **kwargs):
    """
    Computes adjacency matrix of a K-NN graph from the given data.
    :param X: rank 1 np.array, the 2D coordinates of pixels on the grid.
    :param kwargs: kwargs for sklearn.neighbors.kneighbors_graph (see docs
    [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html)).
    :return: scipy sparse matrix.
    """
    A = kneighbors_graph(X, k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))

    return A

def _mnist_img_grid_graph(in_img, k, threshold=0.5):
    """
    Get the adjacency matrix for the KNN graph.
    :param k: int, number of neighbours for each node;
    :return:
    """
    X = _grid_coordinates_from_img(in_img, threshold=threshold)
    A = _get_adj_from_data(
        X, k, mode='distance', metric='euclidean', include_self=False
    )
    return A, X

def draw_graph_mpl(g, pos=None, ax=None, layout_func=nx.drawing.layout.kamada_kawai_layout):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = None
    if pos is None:
        pos = layout_func(g)
    node_color = []
    node_labels = {}
    shift_pos = {}
    for k in g:
        node_color.append(g.nodes[k].get('color', 'green'))
        node_labels[k] = g.nodes[k].get('label', k)
        shift_pos[k] = [pos[k][0], pos[k][1]]

    edge_color = []
    edge_width = []
    for e in g.edges():
        edge_color.append(g.edges[e].get('color', 'black'))
        edge_width.append(g.edges[e].get('width', 0.5))
    nx.draw_networkx_edges(g, pos, edge_color=edge_color, width=edge_width, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_shape='p', node_size=10, alpha=0.75)
    ax.autoscale()
    return fig, ax, pos

def _package_images(img_stack, k, max_nodes):

    X_feat = np.zeros((img_stack.shape[0], max_nodes, 3), dtype='float32')
    X_adj = np.zeros((img_stack.shape[0], max_nodes, max_nodes), dtype='float32')
    for i, x in enumerate(tqdm(img_stack)):
        adj_vec, feat_vec = _mnist_img_grid_graph(x/255.0, k)
        c_len = min(feat_vec.shape[0], max_nodes)
        X_feat[i, :c_len] = feat_vec[:c_len]
        X_adj[i, :c_len, :c_len] = adj_vec.todense()[:c_len, :c_len]
    return X_feat, [sp.csr_matrix(x) for x in X_adj] # list of sparse matrices

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(y_train.shape)

x_tot = np.append(X_train, X_test, axis=0)
y_tot = np.append(y_train, y_test, axis=0)
print(x_tot.shape, y_tot.shape)

x_feat, x_adj = _package_images(x_tot, k=8, max_nodes=200)

dir_npz = "dataset_graph_mnist"
dir_png = "dataset_graph_mnist_img"
make_dir(path=dir_npz, refresh=True)
make_dir(path=dir_png, refresh=True)

for idx, _ in enumerate(x_feat):

    make_dir(path=os.path.join(dir_npz, "%s" %(y_tot[idx])))
    make_dir(path=os.path.join(dir_png, "graph_%s" %(y_tot[idx])))
    make_dir(path=os.path.join(dir_png, "matrix_%s" %(y_tot[idx])))

    G = nx.from_scipy_sparse_matrix(x_adj[idx])
    fig, ax, pos = draw_graph_mpl(G)
    fig.savefig(os.path.join(dir_png, "graph_%s" %(y_tot[idx]), "graph_%08d.png" %(idx)))

    plt.figure(figsize=(6, 6))
    plt.imshow(x_adj[idx].todense(), cmap='gray')
    plt.savefig(os.path.join(dir_png, "matrix_%s" %(y_tot[idx]), "graph_%08d.png" %(idx)))

    np.savez_compressed(os.path.join(dir_npz, "%s" %(y_tot[idx]), "graph_%08d" %(idx)), \
        feature=x_feat[idx], \
        adjacency=x_adj[idx], \
        label=y_tot[idx])
