import argparse, os, shutil
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

def _grid_coordinates_from_img(in_img):
    """
    Returns 2D coordinates for a square grid of equally spaced nodes.
    :param side: int, the side of the grid (i.e., the grid has side * side nodes).
    :return: np.array of shape (side * side, 2).
    """

    # in_img = in_img / 255

    x = np.linspace(0, 1, in_img.shape[0], dtype=np.float32)
    y = np.linspace(0, 1, in_img.shape[1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    z = np.stack([
        xx.ravel(),
        yy.ravel(),
        in_img.ravel(),
    ], -1)
    return z

def _get_adj_from_data(X, k, **kwargs):
    """
    Computes adjacency matrix of a K-NN graph from the given data.
    :param X: rank 1 np.array, the 2D coordinates of pixels on the grid.
    :param kwargs: kwargs for sklearn.neighbors.kneighbors_graph (see docs
    [here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html)).
    :return: scipy sparse matrix.
    """
    A = kneighbors_graph(X, k, mode='distance', metric='euclidean', include_self=False).toarray()
    A = sp.csr_matrix(np.maximum(A, A.T))

    return A

def _mnist_img_grid_graph(in_img, k):
    """
    Get the adjacency matrix for the KNN graph.
    :param k: int, number of neighbours for each node;
    :return:
    """
    X = _grid_coordinates_from_img(in_img)
    A = _get_adj_from_data(X, k)
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
    print("Graph generation")
    for i, x in enumerate(tqdm(img_stack)):
        adj_vec, feat_vec = _mnist_img_grid_graph(x/255.0, k)
        c_len = min(feat_vec.shape[0], max_nodes)
        X_feat[i, :c_len] = feat_vec[:c_len]
        X_adj[i, :c_len, :c_len] = adj_vec.todense()[:c_len, :c_len]
    return X_feat, [sp.csr_matrix(x) for x in X_adj]

def main():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    x_tot = np.append(X_train, X_test, axis=0)
    y_tot = np.append(y_train, y_test, axis=0)
    x_feat, x_adj = _package_images(x_tot, k=8, max_nodes=784)

    dir_npz = FLAGS.save_dir
    dir_png = "%s_img" %(dir_npz)
    make_dir(path=dir_npz, refresh=True)
    if(FLAGS.save_img): make_dir(path=dir_png, refresh=True)

    print("Save dataset")
    for idx, _ in enumerate(tqdm(x_feat)):

        make_dir(path=os.path.join(dir_npz, "%s" %(y_tot[idx])))
        make_dir(path=os.path.join(dir_png, "graph_%s" %(y_tot[idx])))
        make_dir(path=os.path.join(dir_png, "matrix_%s" %(y_tot[idx])))

        if(FLAGS.save_img):
            G = nx.from_scipy_sparse_matrix(x_adj[idx])

            fig, ax, pos = draw_graph_mpl(G)
            fig.savefig(os.path.join(dir_png, "graph_%s" %(y_tot[idx]), "graph_%08d.png" %(idx)))

            plt.figure(figsize=(6, 6))
            plt.imshow(x_adj[idx].todense(), cmap='gray')
            plt.savefig(os.path.join(dir_png, "matrix_%s" %(y_tot[idx]), "graph_%08d.png" %(idx)))

        np.savez_compressed(os.path.join(dir_npz, "%s" %(y_tot[idx]), "graph_%08d" %(idx)), \
            feature=x_feat[idx], \
            adjacency=x_adj[idx].toarray(), \
            label=y_tot[idx])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_img', type=int, default=0, help='')
    parser.add_argument('--save_dir', type=str, default='dataset_graph_mnist', help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
