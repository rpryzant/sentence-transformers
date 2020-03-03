import sys
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans, SpectralClustering


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


def treat_clustering(emb1, emb2, npc, k):
    # treating as sperate: 0.7252247783326363, npc=2 k=5
    # concatting and then doing svd/clustering on the parts: 0.7364172032507696 (same args)

    # cluster vectors
    embs = np.concatenate((emb1, emb2), axis=1)
    kmeans = KMeans(n_clusters=k).fit(embs)

    cluster_embs = [None] * k
    for i, (emb, cluster) in enumerate(zip(embs, kmeans.labels_)):
        if cluster_embs[cluster] is None:
            cluster_embs[cluster] = [emb]
        else:
            cluster_embs[cluster].append(emb)

    # try spectral, kmeans


    # svd per cluster
    cluster_pcs = [None] * k
    for cluster, embs in enumerate(cluster_embs):
        cluster_pcs[cluster] = compute_pc(np.array(embs), npc)

    #embs = np.concatenate((emb1, emb2), axis=0)
    #pc = compute_pc(embs, NPC)

    # project each cluster + remove pcs
    # try matching clusters, 1 pc per cluster
    final_emb1 = []
    final_emb2 = []
    for e1, e2 in zip(emb1, emb2):
        # e1c = kmeans.predict([e1])[0]
        # e2c = kmeans.predict([e2])[0]
        # e1 = remove_pc(e1, npc=npc, pc=cluster_pcs[e1c])
        # e2 = remove_pc(e2, npc=npc, pc=cluster_pcs[e2c])
        # final_emb1.append(e1)
        # final_emb2.append(e2)

        e1e2 = np.concatenate(([e1], [e2]), axis=1)
        e1e2c = kmeans.predict(e1e2)[0]
        e1e2 = np.squeeze(remove_pc(e1e2, npc=npc, pc=cluster_pcs[e1e2c]))
        final_emb1.append(e1e2[:768])
        final_emb2.append(e1e2[768:])

    emb1 = np.squeeze(np.array(final_emb1))
    emb2 = np.squeeze(np.array(final_emb2))

    return emb1, emb2


def treat_pc(emb1, emb2, npc):
    embs = np.concatenate((emb1, emb2), axis=0)
    pc = compute_pc(embs, npc)
    emb1 = remove_pc(emb1, npc=npc, pc=pc)
    emb2 = remove_pc(emb2, npc=npc, pc=pc)
    return emb1, emb2


input = sys.argv[1]
NPC = 2
K = 5

f = np.load(open(input, 'rb'))
emb1 = f['emb1']
emb2 = f['emb2']
labels = f['labels']


# emb1, emb2 = treat_pc(emb1, emb2, npc=NPC)
emb1, emb2 = treat_clustering(emb1, emb2, npc=NPC, k=K)


cosine_scores = 1 - paired_cosine_distances(emb1, emb2)
corr = spearmanr(labels, cosine_scores)

print(corr)


