import os

import numpy as np
import networkx as nx
import scipy.sparse as sp
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from sklearn.metrics import pairwise_distances

def node2vec(adj, dataset):
    G = nx.from_scipy_sparse_matrix(adj)
    embedding_path = 'embeddings/{}.txt'.format(dataset)

    if os.path.exists(embedding_path):
        wv = KeyedVectors.load_word2vec_format(embedding_path)
        print('Loaded Node2Vec embeddings from file')
    else:
        print('Running Node2Vec...')
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=30)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        model.wv.save_word2vec_format(embedding_path)
        wv = model.wv

    # Reorder embedding vectors to match the adjacency matrix
    embeddings = np.empty(shape=(len(G.nodes), 64))
    for i in list(G):
        index = wv.key_to_index[str(i)]
        embeddings[i] = wv.vectors[index]

    return pairwise_distances(embeddings, metric='cosine')

def neighborhood_distance_matrix(adj, preds, dataset):
    path = 'distances/{}.npy'.format(dataset)

    if os.path.exists(path):
        dist_matrix = np.load(path)
        print('Loaded neighborhood distance matrix from file')
        return dist_matrix

    # Add self-loops
    G = nx.from_scipy_sparse_matrix(adj + sp.eye(adj.shape[0]))

    preds = np.argmax(preds, axis=1)
    
    dist_matrix = np.zeros((len(G), len(G)))
    for i in range(len(G)):
        for j in range(i):
            dist = _neighborhood_distance(G, i, j, preds)
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist

        if i % 100 == 0:
            print('calculating distances for node {}'.format(i))

    np.save(path, dist_matrix)
    return dist_matrix
    


def _neighborhood_distance(G, i, j, preds):

    dist = 0
    for k in G.neighbors(i):
        for l in G.neighbors(j):
            if preds[k] != preds[l]:
                dist += 1
    
    dist /= len(list(G.neighbors(i))) * len(list(G.neighbors(j)))
    return dist


def weighted_distance_matrix(d1, d2):
    max_d1 = np.amax(d1)
    max_d2 = np.amax(d2)

    delta = max_d1 / max_d2
    print('Using weight {} for the d2 matrix'.format(delta))
    return d1 + delta * d2