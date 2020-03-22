# python sandbox.py bert_embs
# python sandbox.py glove_embs

import sys
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
import os
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from tqdm import tqdm
import random
import itertools
import math

VOCAB = [l.strip() for l in open('bert.vocab')]

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_, svd.explained_variance_ratio_

def remove_pc(X, npc=1, pc=None, explained_var=None):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    if pc is None:
        pc, explained_var = compute_pc(X, npc)

    # weight contribution by variance
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
        # XX = X - X.dot(pc.transpose()) * (pc * np.expand_dims(explained_var, 1))
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
        # XX = X - X.dot(pc.transpose()).dot((pc * np.expand_dims(explained_var, 1)))

    # standard
    # if npc==1:
    #     XX = X - X.dot(pc.transpose()) * pc
    # else:
    #     XX = X - X.dot(pc.transpose()).dot(pc)

    return XX


def treat_clustering(emb1, emb2, npc, k, strategy='neither'):
    # strategy = when to glue vectors: [neither, both, cluster]  
    assert strategy in {'neither', 'both', 'cluster'}

    # cluster vectors
    if strategy in {'both', 'cluster'}:
        embs = np.concatenate((emb1, emb2), axis=1)
    else:
        embs = np.concatenate((emb1, emb2), axis=0)

    kmeans = KMeans(n_clusters=k).fit(embs)
    # kmeans = AffinityPropagation().fit(embs)
    # print(kmeans.labels_); quit()
    # print(kmeans)
    # kmeans = SpectralClustering(n_clusters=k).fit(embs)

    cluster_embs = [None] * k
    for i, (emb, cluster) in enumerate(zip(embs, kmeans.labels_)):
        if cluster_embs[cluster] is None:
            cluster_embs[cluster] = [emb]
        else:
            cluster_embs[cluster].append(emb)



    # svd per cluster
    cluster_pcs = [None] * k
    for cluster, emb_group in enumerate(cluster_embs):
        if strategy == 'cluster':
            emb_group = list(np.array(emb_group)[:, :768]) + list(np.array(emb_group)[:, 768:])

        cluster_pcs[cluster] = compute_pc(np.array(emb_group), npc)

    # project each cluster + remove pcs
    # try matching clusters, 1 pc per cluster
    final_emb1 = []
    final_emb2 = []
    for e1, e2 in zip(emb1, emb2):
        if strategy in {'cluster', 'both'}:
            e1e2 = np.concatenate(([e1], [e2]), axis=1)
            e1c = e2c = kmeans.predict(e1e2)[0]
        else:
            e1c = kmeans.predict([e1])[0]
            e2c = kmeans.predict([e2])[0]


        if strategy == 'both':
            e1e2 = np.concatenate(([e1], [e2]), axis=1)
            e1e2 = remove_pc(e1e2, npc=npc, pc=cluster_pcs[e1c])
            e1 = e1e2[:, :768]
            e2 = e1e2[:, 768:]
        else:
            # don't handle case (not glue_clustering and glue_svd) because 
            # you have no way of grabbing the cluster assignments
            e1 = remove_pc(e1, npc=npc, pc=cluster_pcs[e1c])
            e2 = remove_pc(e2, npc=npc, pc=cluster_pcs[e2c])

        final_emb1.append(e1)
        final_emb2.append(e2)

    emb1 = np.squeeze(np.array(final_emb1))
    emb2 = np.squeeze(np.array(final_emb2))

    return emb1, emb2


def get_word_weights(word_freqs, doc_freqs, a=1e-3,
                     mode='tfidf', normalize=False):

  word_freq_lines = open(word_freqs).readlines()
  doc_freq_lines = open(word_freqs).readlines()

  num_words = int(word_freq_lines[0])
  num_docs = int(doc_freq_lines[0])

  word_freqs = dict([tuple(line.strip().split("\t")) for line in word_freq_lines[1:]])
  doc_freqs = dict([tuple(line.strip().split("\t")) for line in doc_freq_lines[1:]])
  s = 0
  out = {}
  for word, freq in word_freqs.items():
    freq = float(freq)
    s += freq

    tf = math.log(float(freq) + 1.0)
    inv_freq = a / (a + (freq / num_words) )
    idf = math.log((num_docs + 1) / float(freq))

    if mode == 'tfidf':
      out[word] = tf * idf
    elif mode == 'idf':
      out[word] = idf
    elif mode == 'tf':
      out[word] = tf
    elif mode == 'sif':
      out[word] = inv_freq

    # Words in the vocab that are not in the doc_frequencies file get a frequency of 1
    if mode == 'tfidf':
      unknown_word_weight = math.log(1.0) * math.log(num_docs / 1)
    elif mode == 'idf':
      unknown_word_weight = 0.01
    elif mode == 'tf':
      unknown_word_weight = math.log(1.0)
    elif mode == 'sif':
      unknown_word_weight = 0.01

  if normalize:
    for k, v in out.items():
      out[k] = v * 1.0 / s
    unknown_word_weight = 1.0 / s

  return out, unknown_word_weight




def treat_pc(emb1, emb2, npc):
    embs = np.concatenate((emb1, emb2), axis=0)
    pc = compute_pc(embs, npc)
    emb1 = remove_pc(emb1, npc=npc, pc=pc)
    emb2 = remove_pc(emb2, npc=npc, pc=pc)
    return emb1, emb2





import sys

tgt_root = sys.argv[1]








word2weight, unkWeight = get_word_weights(
    '../aux/wikipedia_word_frequencies.txt',
    '../aux/wikipedia_doc_frequencies.txt',
    mode=sys.argv[2],
    normalize=bool(sys.argv[3]))

id2weight = {}
for i, l in enumerate(open('bert.vocab')):
    id2weight[i] = word2weight.get(l.strip(), unkWeight)

# PRE-CENTERING DOMAIN SHIFT
# python sandbox.py raw_embs_ids
NPC = 1
K = -1
strategy = 'no clustering'



train = np.load(open(os.path.join(tgt_root, 'train.embs'), 'rb'), allow_pickle=True)
tokemb1 = train['tokemb1']
tokemb2 = train['tokemb2']
train_ids = np.load(open(os.path.join(tgt_root, 'train.ids'), 'rb'), allow_pickle=True)



to_center = []
for seq in list(tokemb1) + list(tokemb2):
    for elem in seq:
        # skip once hit paddings
        if elem[0] == 0.0:
            break
        else:
            to_center.append(elem)
train_pc, train_explained_var = compute_pc(np.array(to_center), 1)


def center_and_mean(seq, center=None, explained_var=None, ids=None):
    pc = center
    # if center is not None:
    #     pc, mu = center
    l = 0
    for i in range(len(seq)):
        if seq[i][0] != 0:
#            seq[i] = seq[i] - mu
            seq[i] = remove_pc(seq[i], npc=1, pc=pc, explained_var=explained_var)
            l += 1

    weights = np.array([[id2weight.get(i, unkWeight)] for i in ids])
    weighted_seq = np.multiply(weights, seq)
    return np.sum(weighted_seq, axis=0) / l

    # print(weights); quit()
    # return np.prod(seq, axis=0)**(1.0/l) # geometric mean
    # return np.sum(seq[1:], axis=0) / (l - 1)


for NPC in range(70):
    NPC = 40
    train = np.load(open(os.path.join(tgt_root, 'train.embs'), 'rb'), allow_pickle=True)
    emb1 = train['emb1']
    emb2 = train['emb2']
    labels = train['labels']
    embs = np.concatenate((emb1, emb2), axis=0)
    pc, explained_var = compute_pc(embs, NPC)

    correlations = []
    for f in os.listdir(tgt_root):
        if not f.endswith('.embs') or 'train' in f:
            continue

        tgt = np.load(open(os.path.join(tgt_root, f), 'rb'), allow_pickle=True)
        tokemb1 = tgt['tokemb1']
        tokemb2 = tgt['tokemb2']

        tgt_ids = np.load(
            open(os.path.join(tgt_root, f.replace('.embs', '.ids')), 'rb'), 
            allow_pickle=True)
        id1 = tgt_ids['in1']
        id2 = tgt_ids['in2']

        emb1 = []
        emb2 = []
        for seqA, seqB, i1, i2 in zip(tokemb1, tokemb2, id1, id2):
            emb1.append(center_and_mean(seqA, center=train_pc, explained_var=train_explained_var, ids=i1))
            emb2.append(center_and_mean(seqB, center=train_pc, explained_var=train_explained_var, ids=i2))
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)

        labels = tgt['labels']

        emb1 = remove_pc(emb1, npc=NPC, pc=pc, explained_var=explained_var)
        emb2 = remove_pc(emb2, npc=NPC, pc=pc, explained_var=explained_var)

        cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
        corr = spearmanr(labels, cosine_scores).correlation
        correlations.append(corr)

    print('\t'.join([str(x) for x in 
        [NPC, K, strategy, np.mean(correlations)]]))

    quit()
quit()








# TEST OUT PRE-CENTERING IN-DOMAIN
# NPC = 1
# K = -1
# strategy = 'no clustering'

# centers = {}
# for f in os.listdir(tgt_root):
#     if not f.endswith('.embs') or 'train' in f:
#         continue

#     tgt = np.load(open(os.path.join(tgt_root, f), 'rb'), allow_pickle=True)
#     tokemb1 = tgt['tokemb1']
#     tokemb2 = tgt['tokemb2']

#     # get center vecs (mu or pc)
#     to_center = []
#     for seq in list(tokemb1) + list(tokemb2):
#         for elem in seq:
#             # skip once hit paddings
#             if elem[0] == 0.0:
#                 break
#             else:
#                 to_center.append(elem)
#     pc = compute_pc(np.array(to_center), 1)
#     mu = np.mean(np.array(to_center), axis=0)
#     centers[f] = (pc, mu)
# print('done with centers')

# def center_and_mean(seq, center=None):
#     if center is not None:
#         pc, mu = center
#     l = 0
#     for i in range(len(seq)):
#         if seq[i][0] != 0:
# #            seq[i] = seq[i] - mu
#             seq[i] = remove_pc(seq[i], npc=1, pc=pc)
#             l += 1
#     return np.sum(seq, axis=0) / l

# for NPC in range(70):
#     correlations = []
#     for f in os.listdir(tgt_root):
#         if not f.endswith('.embs') or 'train' in f:
#             continue

#         tgt = np.load(open(os.path.join(tgt_root, f), 'rb'), allow_pickle=True)
#         tokemb1 = tgt['tokemb1']
#         tokemb2 = tgt['tokemb2']

#         emb1 = []
#         emb2 = []
#         for seqA, seqB in zip(tokemb1, tokemb2):
#             emb1.append(center_and_mean(seqA, center=centers[f]))
#             emb2.append(center_and_mean(seqB, center=centers[f]))
#         emb1 = np.array(emb1)
#         emb2 = np.array(emb2)

#         labels = tgt['labels']

#         emb1, emb2 = treat_pc(emb1, emb2, npc=NPC)
#         # emb1, emb2 = treat_clustering(emb1, emb2, npc=NPC, k=K, strategy=strategy)
#         cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
#         corr = spearmanr(labels, cosine_scores).correlation
#         correlations.append(corr)

#     print('\t'.join([str(x) for x in 
#         [NPC, K, strategy, np.mean(correlations)]]))

# quit()







# # TEST OUT DOMAIN SHIFT (no clusters)
# NPC = 1
# K = -1
# strategy = 'no clustering'
# for NPC in range(70):
#     train = np.load(open(os.path.join(tgt_root, 'train.embs'), 'rb'))
#     embs = train['embs']
#     embs = embs[:3000]
#     # emb2 = train['emb2']
#     labels = train['labels']
#     # embs = np.concatenate((emb1, emb2), axis=0)
#     pc = compute_pc(embs, NPC)

#     correlations = []
#     # for f in tqdm(os.listdir('.')):
#     for f in os.listdir(tgt_root):
#         if not f.endswith('.embs') or 'train' in f:
#             continue
#         f = np.load(open(os.path.join(tgt_root, f), 'rb'))
#         emb1 = f['emb1']
#         emb2 = f['emb2']
#         labels = f['labels']
#         emb1 = remove_pc(emb1, npc=NPC, pc=pc)
#         emb2 = remove_pc(emb2, npc=NPC, pc=pc)
#         cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
#         corr = spearmanr(labels, cosine_scores).correlation
#         correlations.append(corr)
#     print('\t'.join([str(x) for x in 
#         [NPC, K, strategy, np.mean(correlations)]]))


# quit()







# HYPERPARAM SEARCH (WITH CLUSTERS)

# replicates = sorted(
#     itertools.product(*[range(70), range(2, 20), ['neither', 'both', 'cluster']]),
#     key=lambda x: random.random())
# for NPC, K, strategy in tqdm(replicates): 
strategy = K = None
for NPC in range(71):
    correlations = []
    # for f in tqdm(os.listdir('.')):
    for f in os.listdir(tgt_root):
        if not f.endswith('.embs') or 'train' in f:
            continue
        f = np.load(open(os.path.join(tgt_root, f), 'rb'))
        emb1 = f['emb1']
        emb2 = f['emb2']
        labels = f['labels']
        emb1, emb2 = treat_pc(emb1, emb2, npc=NPC)
        # emb1, emb2 = treat_clustering(emb1, emb2, npc=NPC, k=K, strategy=strategy)
        cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
        corr = spearmanr(labels, cosine_scores).correlation
        correlations.append(corr)
    print('\t'.join([str(x) for x in 
        [NPC, K, strategy, np.mean(correlations)]]))




