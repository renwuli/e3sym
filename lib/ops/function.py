import os
import os.path as osp
import math
import glob
import numpy as np
import torch
import torch.nn as nn

from torch.utils.cpp_extension import load

import numba

print(">>> loading op")
cur_dir = osp.dirname(osp.abspath(__file__))
ops = load(
    name="ops",
    sources=glob.glob(f"{cur_dir}/src/*.cpp") + glob.glob(f"{cur_dir}/src/*.cu"),
    extra_include_paths=[f"{cur_dir}/include"],
    build_directory=".tmp/",
)


@numba.jit(nopython=True)
def find_connected_component(neighbours, min_cluster_size):
    num_points = int(neighbours.shape[0])
    visited = np.zeros((num_points,), dtype=numba.types.bool_)
    clusters = []
    for i in range(num_points):
        if visited[i]:
            continue

        cluster = []
        queue = []
        visited[i] = True
        queue.append(i)
        cluster.append(i)

        while len(queue):
            k = queue.pop()
            k_neighbours = neighbours[k]
            for nei in k_neighbours:
                if nei.item() == -1:
                    break

                if not visited[nei]:
                    visited[nei] = True
                    queue.append(nei.item())
                    cluster.append(nei.item())

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def get_clusters(pos, batch, thre, nsample, min_cluster_size):
    """
    pos:        [N, 3]
    batch:      [N]
    """
    assert pos.shape[0] == batch.shape[0]
    graph = ops.build_graph_batch(pos, pos, batch, batch, thre, nsample)
    neighbours = graph[0].cpu().numpy()
    edges = graph[2]
    clusters = find_connected_component(neighbours, min_cluster_size)

    return clusters, edges


def extract_cluster_centriod(pos, edge, batch, clusters):
    """
    pos:        [N, 3], tensor
    edge:       [N, K], tenor
    batch:      [N], tensor
    clusters:   [M], list
    """
    assert pos.shape[0] == batch.shape[0]
    num_clusters = len(clusters)
    if 0 == num_clusters:
        return torch.LongTensor([0])

    indices = []
    for i, cluster in enumerate(clusters):
        idx = cluster[torch.argmax(edge[cluster].sum(-1))]
        indices.append(idx)

    indices = torch.LongTensor(indices)
    return indices
