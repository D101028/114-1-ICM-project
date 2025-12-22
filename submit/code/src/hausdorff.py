import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple

from .cluster import ClusterGroup

def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the hausdorff distance between two sets of points A and B.
    """
    # A, B shape: (N, 2), (M, 2)
    if A.size == 0 or B.size == 0:
        return np.inf # 回傳最大值

    # 使用 cdist 計算成對距離矩陣 (N, M)
    dists = cdist(A, B, metric='euclidean')
    
    # Directed distances
    h_A_B = np.max(np.min(dists, axis=1))
    h_B_A = np.max(np.min(dists, axis=0))
    
    return max(h_A_B, h_B_A)

def hausdorff_similarity(base_cluster: ClusterGroup, cluster: ClusterGroup) -> float:
    """
    計算兩個 ClusterGroup 的 hausdorff 相似度。
    
    :param base_cluster: templates cluster
    :type base_cluster: ClusterGroup
    :param cluster: 欲比較之 cluster，先縮放到與 base_cluster 相同 size 再比較
    :type cluster: ClusterGroup
    :return: [0, 1] 區間的值
    :rtype: float
    """
    # resize 
    h, w = base_cluster.get_bbox_hw()
    cluster = cluster.resize((w, h))

    arr1 = base_cluster.get_pixels_below_threshold()
    arr2 = cluster.get_pixels_below_threshold()

    if arr1.size == 0 or arr2.size == 0:
        return 0.0 # 若為空，返回 0 相似

    dist = hausdorff_distance(arr1, arr2)
    
    # 使用兩者聯集的對角線進行標準化 (最推薦，確保 ratio <= 1)
    all_pts = np.vstack([arr1, arr2])
    min_coords = all_pts.min(axis=0)
    max_coords = all_pts.max(axis=0)
    diff = max_coords - min_coords
    diagonal = np.sqrt(np.sum(diff**2))
    
    # 避免除以零
    if diagonal == 0:
        return 0.0 if dist == 0 else 1.0
        
    ratio = dist / diagonal
    return 1 - ratio
