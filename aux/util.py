
from collections import defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import random
import sys; sys.path.append('../sentence_transformers')
from sentence_transformers.util import batch_to_device
from tqdm import tqdm

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

def remove_pc(X, pc=None):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    if pc is None:
        pc = compute_pc(X, npc)

    npc = pc.shape[0]

    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def embed_dataloader(dataloader, model, sample_size=-1, data_size=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader.collate_fn = model.smart_batching_collate
    labels = []

    # if sample_size > 0 and data_size > 0:
    #     sample_rate = sample_size * 1.0 / data_size

    embs1 = []
    embs2 = []
    tok_embs1 = []
    tok_embs2 = []
    input_ids1 = []
    input_ids2 = []

    for step, batch in tqdm(enumerate(dataloader)):
        features, label_ids = batch_to_device(batch, device)
        labels.extend(label_ids.to("cpu").numpy())

        with torch.no_grad():
            emb1, emb2 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in features]
            tok_emb1, tok_emb2 = [model(sent_features)['tok_embs'].to("cpu").numpy() for sent_features in features]
            
            in1, in2 = [sent_features['input_ids'].to('cpu').numpy() for sent_features in features]


        embs1 += list(emb1)
        embs2 += list(emb2)
        tok_embs1 += list(tok_emb1)
        tok_embs2 += list(tok_emb2)
        input_ids1 += list(in1)
        input_ids2 += list(in2)

        if len(embs1) + len(embs2) > sample_size:
            break

    # tok_embs* will be ragged in 2nd dim (padded seq len)
    return np.array(tok_embs1), np.array(tok_embs2), np.array(embs1), np.array(embs2), labels, np.array(input_ids1), np.array(input_ids2)






