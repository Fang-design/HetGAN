import numpy as np
import torch
import math


def normalize(A,symmetric=True):
    A = A + torch.eye(A.size(0))
    d = A.sum(1)
    if symmetric:
        D = torch.diag(torch.pow(d,-0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d,-1))
        return D.mm(A)






def normalization(dataset):
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    dataset = dataset.reshape(-1,1)
    scalar = max_value - min_value
    dataset = list(map(lambda x: (x-min_value) / scalar, dataset))
    return dataset


def get_laplacian(W):
    d = W.sum(1)
    D = torch.diag(d)
    L = D - W
    return L

def transform(d):
    d = d.numpy()
    sigma = np.std(d)
    epsilon = 0
    w = np.zeros([d.shape[0],d.shape[1]])
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            item = d[i][j] * d[i][j]/sigma/sigma
            if i != j and math.exp(-item) >= epsilon:
                w[i][j] = math.exp(-item)
            else:
                w[i][j] = 0
    w = torch.from_numpy(w)
    w = w.to(torch.float32)
    return w

