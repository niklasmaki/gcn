import numpy as np
import networkx as nx
import hnswlib
from gcn.distances import node2vec, node2vec_distances


def get_allowed_edges(adj, dataset):
    embeddings = node2vec(adj, dataset)
    print(embeddings.shape)

    num_elements, dim = embeddings.shape
    data_labels = np.arange(num_elements)

    p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip
    p.init_index(max_elements = num_elements, ef_construction = 500, M = 64)

    p.add_items(embeddings, data_labels)

    p.set_ef(500)
    print('starting knn search')
    labels, distances = p.knn_query(embeddings, k = 6)
    print('got results:')
    print(labels)
    print(distances)
    return labels



def test():
    G = nx.erdos_renyi_graph(20, 0.1, seed=1)
    adj = nx.adjacency_matrix(G)
    D = node2vec_distances(adj, "test")
    print("Running MAP estimation")
    w = map_estimate(adj, D, "test")

    print("done")
    print(w.shape)
    #print(w)

    


def map_estimate(adj, D, dataset):

    m = adj.shape[0]
    w = vech(adj)
    z = vech(D)

    # Make dimensions match
    z = z[..., np.newaxis]
    w = w[..., np.newaxis]

    d = np.sum(adj, axis=1)

    m = len(d)

    labels = get_allowed_edges(adj, dataset)
    
    allowed_adj = np.zeros((m, m))
    for row_idx in range(len(labels)):
        row = labels[row_idx]
        for col_idx in row:
            if row_idx in labels[col_idx]: # Check that both of the nodes belong to each other's nearest neighbors
                allowed_adj[row_idx, col_idx] = 1
                allowed_adj[col_idx, row_idx] = 1

    allowed_adj_vech_mask = vech(allowed_adj) > 0

    w_allowed = w[allowed_adj_vech_mask]
    z_allowed = z[allowed_adj_vech_mask]


    #S = _generate_S(m)
    S = _generate_S_allowed(m, allowed_adj_vech_mask)
    
    #_verify_S(S, w_allowed, adj)

    result = primal_dual(S, z_allowed, 0.5, 0.1, w_allowed, d, 0.05, 0.02)
    result = np.asarray(result).squeeze()
    #result = result.round(decimals=1)
    #result = np.ceil(result).astype(int)
    #result = np.reshape(result, (1,35))
    print(result.shape)

    full_result_vech = np.zeros(len(w))
    full_result_vech[allowed_adj_vech_mask] = result[0]
    print(full_result_vech.shape)

    full_result = np.zeros(adj.shape)
    full_result[np.triu_indices(m, k=1)] = full_result_vech
    print(full_result.shape)


    return full_result

def vech(A):
    m = A.shape[0]
    vech = A[np.triu_indices(m, k=1)]
    return np.squeeze(np.asarray(vech))



def primal_dual(S, z, alpha, beta, w, d, step_size, tolerance):
    """Primal dual algorithm for estimating the MAP solution.

    Args:
        S (np.array): A linear operator that satisfies "Sw = array of degrees of the nodes"
        z (np.array): The half-vectorization of the distance matrix Z, length m*(m-1)/2
        alpha (float): Controls the scale of the solution
        beta ([type]): Controls the sparsity of the solution
        w (np.array): The half-vectorization of an edge weight matrix, length m*(m-1)/2
        d (np.array): Unknown, length m
        step_size (float): Step size of the algorithm
        tolerance (float): Controls when to stop the algorithm
    """
    iterations = 100
    for i in range(iterations):
        print('---------------------- Iteration', i)
        prev_w = w
        prev_d = d
        y = w - step_size * (2*beta*w + np.dot(S.T, d))
        y_bar = d + step_size * np.dot(S, w)
        p = np.maximum(0, y - 2*step_size*z)
        p_bar = 0.5*(y_bar - np.sqrt(np.square(y_bar) + 4*alpha*step_size))
        q = p - step_size * (2*beta*p + np.dot(S.T, p_bar))
        q_bar = p_bar + step_size * np.dot(S, p)
        w = w - y + q
        d = d - y_bar + q_bar
        
        if np.linalg.norm(w - prev_w)/np.linalg.norm(prev_w) < tolerance \
            and np.linalg.norm(d - prev_d)/np.linalg.norm(prev_d) < tolerance:
            print("The algorithm has converged")
            return w
    return w

def _generate_S(m):
    shape = (m, int(m*(m-1)/2))
    S = np.zeros(shape)

    col = 0
    for i in range(m):
        for j in range(i+1, m):
            S[i, col] = 1
            S[j, col] = 1
            col += 1
    return S

def _generate_S_allowed(m, mask):
    print(mask)
    print(np.sum(mask))
    shape = (m, np.sum(mask))
    S = np.zeros(shape)

    mask_idx = 0
    col = 0
    for i in range(m):
        for j in range(i+1, m):
            if mask[mask_idx]:
                S[i, col] = 1
                S[j, col] = 1
                col += 1
            mask_idx += 1
    return S

def _verify_S(S, w, adj):
    d1 =  np.dot(S, w)
    d2 = np.sum(adj, axis=1)
    d2 = np.squeeze(np.asarray(d2))
    if not np.array_equal(d1, d2):
        print('Problem with generating S! The arrays Sw and d are not equal!')
        print(d1)
        print(d2)
        raise Exception("Problem with generating S") 