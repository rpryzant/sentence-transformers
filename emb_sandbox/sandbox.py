import sys
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
import os
from sklearn.cluster import KMeans, SpectralClustering
from tqdm import tqdm
import random
import itertools

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

    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
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








def treat_pc(emb1, emb2, npc):
    embs = np.concatenate((emb1, emb2), axis=0)
    pc = compute_pc(embs, npc)
    emb1 = remove_pc(emb1, npc=npc, pc=pc)
    emb2 = remove_pc(emb2, npc=npc, pc=pc)
    return emb1, emb2





# neither
# 5 5 0.7351470132727655
# both
# 5 5 0.7333437322101058
# cluster only
# 5 5 0.7403271596663675

replicates = sorted(
    itertools.product(*[range(70), range(2, 20), ['neither', 'both', 'cluster']]),
    key=lambda x: random.random())

for NPC, K, strategy in tqdm(replicates): 
    correlations = []

    # for f in tqdm(os.listdir('.')):
    for f in os.listdir('.'):
        if not f.endswith('.embs') or 'train' in f:
            continue

        f = np.load(open(f, 'rb'))
        emb1 = f['emb1']
        emb2 = f['emb2']
        labels = f['labels']


        # emb1, emb2 = treat_pc(emb1, emb2, npc=NPC)
        emb1, emb2 = treat_clustering(emb1, emb2, npc=NPC, k=K, strategy=strategy)

        cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
        corr = spearmanr(labels, cosine_scores).correlation
        correlations.append(corr)

    print('\t'.join([str(x) for x in 
        [NPC, K, strategy, np.mean(correlations)]]))




