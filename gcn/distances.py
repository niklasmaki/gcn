import os

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from sklearn.metrics import pairwise_distances

def node2vec(adj, dataset):
    G = nx.from_scipy_sparse_matrix(adj)
    embedding_path = 'embeddings/{}.txt'.format(dataset)

    if os.path.exists(embedding_path):
        wv = KeyedVectors.load_word2vec_format(embedding_path)
    else:
        
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
