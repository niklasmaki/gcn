import numpy as np

def test():

    m = 4

    W = np.array([0,2,4,6,2,0,8,10,4,8,0,12,6,10,12,0]).reshape(m,m)
    print(W)
    W_upper = W[np.triu_indices(4, k=1)]
    print(np.triu_indices(4, k=1))
    print(W_upper)

    S = _generate_S(m)



def primal_dual(z, alpha, beta, w, d, step_size, tolerance):
    iterations = 100
    prev_w = w
    prev_d = d
    for i in range(iterations):
        pass
        #y = w - step_size * (2*beta*w + )

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
    print(S)