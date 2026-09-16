"""SoC（AAAI'24）的多粒度 k-means：在类-类转移亲和矩阵上聚类。

聚类距离 = (M[i][c] + M[c][i]) / 2（双向转移计数的均值，类 i 与中心类 c 的亲和度）。
对粒度级 k ∈ {2, ..., K+1} 各聚类一次，得到 K 个由粗到细的
类簇划分——低置信样本用粗粒度（选择集扩张），高置信样本用细粒度（选择集收缩）。
"""

from __future__ import annotations

import numpy as np


def _calc_dis(data_set, centroids, k):
    clalist = []
    temp = []
    for i in range(np.size(data_set, 0)):
        for c in centroids:
            temp.append((data_set[i][c] + data_set[c][i]) / 2.0)
        clalist.append(temp)
        temp = []
    return clalist


def _classify(data_set, centroids, k, reverse):
    clalist = _calc_dis(data_set, centroids, k)
    min_dist_indices = np.argmin(clalist, axis=1) if reverse else np.argmax(clalist, axis=1)
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(min_dist_indices)):
        if x not in cluster[min_dist_indices[x]] and x not in centroids:
            cluster[min_dist_indices[x]].append(x)
    new_centroids = []
    subgraph = []
    for x in cluster:
        temp = [[] for _ in range(len(x))]
        for i in range(len(x)):
            for y in x:
                temp[i].append(data_set[x[i]][y])
        subgraph.append(temp)
    for i in range(k):
        min_value = [np.sum(row) for row in subgraph[i]]
        if reverse:
            new_centroids.append(cluster[i][int(np.argmin(min_value))])
        else:
            new_centroids.append(cluster[i][int(np.argmax(min_value))])
    changed = set(new_centroids) == set(centroids)
    return changed, new_centroids


def kmeans(data_set, k, centroids, reverse=False):
    """在亲和矩阵上做 k-means，返回 (类->簇索引 dict, 簇列表, 新中心)。"""
    changed, new_centroids = _classify(data_set, centroids, k, reverse)
    n = 0
    while not changed and n < 2000:
        changed, new_centroids = _classify(data_set, new_centroids, k, reverse)
        n += 1
    clalist = _calc_dis(data_set, new_centroids, k)
    min_dist_indices = np.argmin(clalist, axis=1) if reverse else np.argmax(clalist, axis=1)
    cluster = [[centroids[i]] for i in range(k)]
    for x in range(len(min_dist_indices)):
        if x not in cluster[min_dist_indices[x]] and x not in centroids:
            cluster[min_dist_indices[x]].append(x)
    dic = {}
    for i, j in enumerate(cluster):
        for x in j:
            dic[x] = i
    return dic, cluster, new_centroids


def multigranularity_kmeans(affinity: np.ndarray, num_granularity: int, reverse=False):
    """对同一亲和矩阵聚类出 num_granularity 个由粗到细的粒度。

    返回 (label_dics, clusters)：第 i 个粒度有 i+2 个簇（与 SoC 原实现一致，
    起始粒度 k=2）。
    """
    label_dics, clusters = [], []
    for i in range(num_granularity):
        k = i + 2
        centroids = list(range(k))
        dic, cluster, _ = kmeans(affinity, k, centroids, reverse=reverse)
        label_dics.append(dic)
        clusters.append(cluster)
    return label_dics, clusters
