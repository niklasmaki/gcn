import numpy as np
import networkx as nx
from gcn.distances import node2vec

def test():
    G = nx.erdos_renyi_graph(20, 0.1)
    adj = nx.adjacency_matrix(G)
    print(adj)
    D = node2vec(adj, "test")
    print("Running MAP estimation")
    w = map_estimate(adj, D)

    print("done")
    print(w.shape)

    


def map_estimate(adj, D):

    m = adj.shape[0]
    w = adj[np.triu_indices(m, k=1)]
    z = D[np.triu_indices(m, k=1)]

    print('m', m)
    print('---')
    print(w)
    print(w.T)
    w = np.squeeze(w)
    print(w[0].shape)
    print('---')
    print(z)

    d = np.sum(adj, axis=1)
    print(d.shape)
    print(d)
    print(d.T)


    return primal_dual(z, 1, 1, w.T, d, 0.05, 0.02)


def primal_dual(z, alpha, beta, w, d, step_size, tolerance):
    """Primal dual algorithm for estimating the MAP solution.

    Args:
        z (np.array): The half-vectorization of the distance matrix Z, length m*(m-1)/2
        alpha (float): Controls the scale of the solution
        beta ([type]): Controls the sparsity of the solution
        w (np.array): The half-vectorization of an edge weight matrix, length m*(m-1)/2
        d (np.array): Unknown, length m
        step_size (float): Step size of the algorithm
        tolerance (float): Controls when to stop the algorithm
    """

    m = len(d)
    S = _generate_S(m)
    print('S', S.shape)
    print(np.linalg.norm(S))

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
        print(w)
        print(d)
        if np.linalg.norm(w - prev_w)/np.linalg.norm(prev_w) < tolerance \
            and np.linalg.norm(d - prev_d)/np.linalg.norm(prev_d) < tolerance:
            print("The algorithm has converged")
            return w

def _generate_S(m):
    shape = (m, int(m*(m-1)/2))
    S = np.zeros(shape)

    col = 0
    for i in range(m):
        for j in range(i+1, m):
            print(i, j, col)
            S[i, col] = 1
            S[j, col] = 1
            col += 1
    return S