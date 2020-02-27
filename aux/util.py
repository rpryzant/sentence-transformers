
from collections import defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import random
import sys; sys.path.append('../sentence_transformers')
from sentence_transformers.util import batch_to_device


def get_tok_weights(model, weightfile, a=1e-3):
    d = defaultdict(float)
    N = 0.0
    for l in open(weightfile):
        word, freq = l.strip().split()
        freq = float(freq)
        for tok in model.tokenize(word):
            d[tok] += freq
            N += freq

    for key, value in d.items():
        d[key] = a / (a + value/N)

    return d



def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in xrange(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1, pc=None):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    if pc is None:
        pc = compute_pc(X, npc)
        print("HERE")
    # print(pc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def embed_dataloader(dataloader, model, sample_size=-1, data_size=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader.collate_fn = model.smart_batching_collate

    if sample_size > 0 and data_size > 0:
        sample_rate = sample_size * 1.0 / data_size

    embs = []

    for step, batch in enumerate(dataloader):
        features, label_ids = batch_to_device(batch, device)
        with torch.no_grad():
            emb1, emb2 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in features]

            if random.random() < sample_rate:
                embs += list(emb1)
            if random.random() < sample_rate:
                embs += list(emb2)

    random.shuffle(embs)

    return np.array(embs)






